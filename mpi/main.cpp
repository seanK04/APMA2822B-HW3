#include <iostream>
#include <vector>
#include <cmath>
#include <sys/time.h>
#include <mpi.h>
#include <string>
#include <sstream>
#include <iomanip>
#include <algorithm> // For std::min

using Grid = std::vector<std::vector<double>>;

struct Params {
    int Nx = 256;
    int Ny = 256;
    int max_iters = 100000;
    double tol = 1e-6;
    int print_every = 1000;
    double Lx = 1.0, Ly = 1.0;
};

static inline double sqr(double x) { return x * x; }

// Argument parsing (serial)
void parse_args(int argc, char* argv[], Params& p) {
    for (int k = 1; k < argc; ++k) {
        std::string a = argv[k];
        auto next = [&](double& target){
            if (k + 1 < argc) { target = std::stod(argv[++k]); }
        };
        auto nexti = [&](int& target){
            if (k + 1 < argc) { target = std::stoi(argv[++k]); }
        };

        if (a == "--nx") nexti(p.Nx);
        else if (a == "--ny") nexti(p.Ny);
        else if (a == "--tol") next(p.tol);
        else if (a == "--max-iters") nexti(p.max_iters);
        else if (a == "--print-every") nexti(p.print_every);
        else if (a == "--lx") next(p.Lx);
        else if (a == "--ly") next(p.Ly);
        else if (a == "--help" || a == "-h") {
            std::cout <<
                "Usage: ./mpi_solver [--nx N] [--ny N] [--tol T] [--max-iters M]\n"
                "                  [--print-every K] [--lx Lx] [--ly Ly]\n";
            std::exit(0);
        }
    }
}

// Grid helper (creates a local grid)
Grid make_grid(int local_Nx, int Ny, double init = 0.0) {
    return Grid(local_Nx, std::vector<double>(Ny, init));
}

// MPI-aware zero_boundary
void zero_boundary(Grid& u, int rank, int num_procs, int local_Nx) {
    const int Ny = (int)u[0].size();

    // All processes zero their local left/right boundaries
    for (int i = 0; i < local_Nx + 2; ++i) {
        u[i][0] = 0.0;
        u[i][Ny-1] = 0.0;
    }

    // Rank 0 zeroes the global top boundary (its first real row)
    if (rank == 0) {
        for (int j = 0; j < Ny; ++j) {
            u[1][j] = 0.0;
        }
    }
    
    // Last rank zeroes the global bottom boundary (its last real row)
    if (rank == num_procs - 1) {
        for (int j = 0; j < Ny; ++j) {
            u[local_Nx][j] = 0.0;
        }
    }
}

// MPI-aware fill_rhs
void fill_rhs(Grid& f, double Lx, double Ly, int local_Nx, int my_row_start, int Nx_global, int Ny_global) {
    const double dx = Lx / (Nx_global - 1);
    const double dy = Ly / (Ny_global - 1);

    for (int i = 0; i < local_Nx; ++i) {
        int global_i = my_row_start + i;
        double x = global_i * dx;
        for (int j = 0; j < Ny_global; ++j) {
            double y = j * dy;
            // Write into local row i+1 (to skip ghost row)
            f[i+1][j] = -8.0 * M_PI * M_PI * std::sin(2.0 * M_PI * x) * std::sin(2.0 * M_PI * y);
        }
    }
}

// One Jacobi sweep (MPI version)
// Returns local L2 difference (sum of squares)
double jacobi_sweep(Grid& u_old, Grid& u_new, const Grid& f, 
                    double dx, double dy, int rank, int num_procs, int local_Nx)
{
    const int Ny = (int)u_old[0].size();
    
    // --- 1. Halo Exchange (non-blocking) ---
    int rank_up = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    int rank_down = (rank < num_procs - 1) ? rank + 1 : MPI_PROC_NULL;

    MPI_Request reqs[4];
    MPI_Status stats[4];

    // Post non-blocking receives
    // Recv from rank_up into top ghost row (u_old[0])
    MPI_Irecv(u_old[0].data(), Ny, MPI_DOUBLE, rank_up, 0, MPI_COMM_WORLD, &reqs[0]);
    // Recv from rank_down into bottom ghost row (u_old[local_Nx+1])
    MPI_Irecv(u_old[local_Nx+1].data(), Ny, MPI_DOUBLE, rank_down, 0, MPI_COMM_WORLD, &reqs[1]);

    // Post non-blocking sends
    // Send our first real row (u_old[1]) to rank_up
    MPI_Isend(u_old[1].data(), Ny, MPI_DOUBLE, rank_up, 0, MPI_COMM_WORLD, &reqs[2]);
    // Send our last real row (u_old[local_Nx]) to rank_down
    MPI_Isend(u_old[local_Nx].data(), Ny, MPI_DOUBLE, rank_down, 0, MPI_COMM_WORLD, &reqs[3]);

    // --- 2. Compute ---
    const double invdx2 = 1.0 / (dx*dx);
    const double invdy2 = 1.0 / (dy*dy);
    const double denom  = 2.0*invdx2 + 2.0*invdy2;

    double local_l2diff_sq = 0.0;

    // Wait for all communications to complete
    MPI_Waitall(4, reqs, stats);

    // boundaries remain zero (Dirichlet)
    // Loop iterates over all REAL local rows (1 to local_Nx)
    for (int i = 1; i <= local_Nx; ++i) {
        for (int j = 1; j < Ny - 1; ++j) {
            const double neighbor_sum =
                (u_old[i-1][j] + u_old[i+1][j]) * invdx2 +
                (u_old[i][j-1] + u_old[i][j+1]) * invdy2;

            const double next_val = (neighbor_sum - f[i][j]) / denom;
            const double d = next_val - u_old[i][j];
            u_new[i][j] = next_val;
            local_l2diff_sq += d*d;
        }
    }
    
    // Enforce boundaries on the new grid
    zero_boundary(u_new, rank, num_procs, local_Nx);
    
    return local_l2diff_sq; // Return local sum of squares
}

// L2 error vs. exact solution (MPI version)
// Returns local L2 error (sum of squares)
double l2_error_vs_exact(const Grid& u, double Lx, double Ly, int local_Nx, int my_row_start, int Nx_global, int Ny_global) {
    const double dx = Lx / (Nx_global - 1);
    const double dy = Ly / (Ny_global - 1);
    double local_acc = 0.0;
    
    for (int i = 0; i < local_Nx; ++i) {
        int global_i = my_row_start + i;
        double x = global_i * dx;
        for (int j = 0; j < Ny_global; ++j) {
            double y = j * dy;
            double u_exact = std::sin(2.0 * M_PI * x) * std::sin(2.0 * M_PI * y);
            // Compare against local row i+1
            local_acc += sqr(u[i+1][j] - u_exact);
        }
    }
    return local_acc;
}

// Pretty printing of params (only rank 0)
void print_header(const Params& p, int num_procs) {
    std::cout << "==== MPI Jacobi (parallel) ====\n"
              << "Grid: " << p.Nx << " x " << p.Ny
              << "   tol: " << p.tol
              << "   max_iters: " << p.max_iters << "\n"
              << "Running with " << num_procs << " MPI processes.\n";
}

int main(int argc, char* argv[]) {
    MPI_Init(&argc, &argv);

    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);

    Params P;
    parse_args(argc, argv, P);
    if (rank == 0) {
        print_header(P, num_procs);
    }

    // Global grid parameters
    const double dx = P.Lx / (P.Nx - 1);
    const double dy = P.Ly / (P.Ny - 1);

    // --- Domain Decomposition ---
    // Calculate 1D split of rows
    int base_rows = P.Nx / num_procs;
    int remainder = P.Nx % num_procs;
    int local_Nx = base_rows + (rank < remainder ? 1 : 0);
    int my_row_start = rank * base_rows + std::min(rank, remainder);
    // ----------------------------

    // Allocate local grids (local_Nx + 2 ghost rows)
    Grid u     = make_grid(local_Nx + 2, P.Ny, 0.0);
    Grid u_new = make_grid(local_Nx + 2, P.Ny, 0.0);
    Grid rhs   = make_grid(local_Nx + 2, P.Ny, 0.0);

    // Initialize local grids
    zero_boundary(u, rank, num_procs, local_Nx);
    zero_boundary(u_new, rank, num_procs, local_Nx);
    fill_rhs(rhs, P.Lx, P.Ly, local_Nx, my_row_start, P.Nx, P.Ny);

    int iter = 0;
    double diff = 1.0;

    struct timeval t0, t1;
    MPI_Barrier(MPI_COMM_WORLD); // Sync all processes before timer
    gettimeofday(&t0, NULL);

    while (iter < P.max_iters && diff > P.tol) {
        double local_l2diff_sq = jacobi_sweep(u, u_new, rhs, dx, dy, rank, num_procs, local_Nx);
        
        // swap roles
        std::swap(u, u_new);

        // Global reduction for convergence check
        double global_l2diff_sq = 0.0;
        MPI_Allreduce(&local_l2diff_sq, &global_l2diff_sq, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

        diff = std::sqrt(global_l2diff_sq);
        ++iter;

        if (rank == 0 && P.print_every > 0 && (iter % P.print_every == 0)) {
            std::cout << "Iter " << std::setw(6) << iter
                      << "  ||u^{k+1}-u^{k}||_2 = " << std::scientific << diff << "\n";
        }
    }

    gettimeofday(&t1, NULL);

    double elapsed = (t1.tv_sec - t0.tv_sec);
    elapsed += (t1.tv_usec - t0.tv_usec) / 1000000.0;

    // Final error vs exact
    double local_err_sq = l2_error_vs_exact(u, P.Lx, P.Ly, local_Nx, my_row_start, P.Nx, P.Ny);
    double global_err_sq = 0.0;
    MPI_Reduce(&local_err_sq, &global_err_sq, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    // Rank 0 prints the final report
    if (rank == 0) {
        double err_exact = std::sqrt(global_err_sq / (P.Nx * P.Ny));
        std::cout << "\n--------------------------------\n";
        std::cout << "Converged iters : " << iter << "\n";
        std::cout << "Last L2 diff    : " << std::scientific << diff << "\n";
        std::cout << "Elapsed (s)     : " << std::fixed << std::setprecision(6) << elapsed << "\n";
        std::cout << "L2 error vs exact: " << std::scientific << err_exact << "\n";
        std::cout << "--------------------------------\n";
    }

    MPI_Finalize();
    return 0;
}