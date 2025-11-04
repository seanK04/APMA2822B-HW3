#include <iostream>
#include <vector>
#include <cmath>
#include <sys/time.h>
#include <string>
#include <sstream>
#include <iomanip>

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

// Argument parsing 
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
                "Usage: ./baseline [--nx N] [--ny N] [--tol T] [--max-iters M]\n"
                "                  [--print-every K] [--lx Lx] [--ly Ly]\n";
            std::exit(0);
        }
    }
}

// Grid helpers
Grid make_grid(int Nx, int Ny, double init = 0.0) {
    return Grid(Nx, std::vector<double>(Ny, init));
}

void zero_boundary(Grid& u) {
    const int Nx = (int)u.size();
    const int Ny = (int)u[0].size();
    for (int i = 0; i < Nx; ++i) {
        u[i][0] = 0.0;
        u[i][Ny-1] = 0.0;
    }
    for (int j = 0; j < Ny; ++j) {
        u[0][j] = 0.0;
        u[Nx-1][j] = 0.0;
    }
}

// Problem setup
// u_xx + u_yy = f(x,y)
// Exact: u = sin(2πx)*cos(2πy)
// f = -8π² sin(2πx) cos(2πy)
void fill_rhs(Grid& f, double Lx, double Ly) {
    const int Nx = (int)f.size();
    const int Ny = (int)f[0].size();
    const double dx = Lx / (Nx - 1);
    const double dy = Ly / (Ny - 1);
    for (int i = 0; i < Nx; ++i) {
        double x = i * dx;
        for (int j = 0; j < Ny; ++j) {
            double y = j * dy;
            f[i][j] = -8.0 * M_PI * M_PI * std::sin(2.0 * M_PI * x) * std::sin(2.0 * M_PI * y);
        }
    }
}

// One Jacobi sweep: compute u_new from u_old and f
// Returns L2 difference ||u_new - u_old||_2
double jacobi_sweep(const Grid& u_old, Grid& u_new,
                    const Grid& f, double dx, double dy)
{
    const int Nx = (int)u_old.size();
    const int Ny = (int)u_old[0].size();

    const double invdx2 = 1.0 / (dx*dx);
    const double invdy2 = 1.0 / (dy*dy);
    const double denom  = 2.0*invdx2 + 2.0*invdy2;

    double l2diff = 0.0;

    // boundaries remain zero (Dirichlet)
    for (int i = 1; i < Nx - 1; ++i) {
        for (int j = 1; j < Ny - 1; ++j) {
            const double neighbor_sum =
                (u_old[i-1][j] + u_old[i+1][j]) * invdx2 +
                (u_old[i][j-1] + u_old[i][j+1]) * invdy2;

            const double next_val = (neighbor_sum - f[i][j]) / denom;
            const double d = next_val - u_old[i][j];
            u_new[i][j] = next_val;
            l2diff += d*d;
        }
    }
    // enforce boundary (in case)
    zero_boundary(u_new);
    return std::sqrt(l2diff);
}

// L2 error vs. exact solution
double l2_error_vs_exact(const Grid& u, double Lx, double Ly) {
    const int Nx = (int)u.size();
    const int Ny = (int)u[0].size();
    const double dx = Lx / (Nx - 1);
    const double dy = Ly / (Ny - 1);
    double acc = 0.0;
    for (int i = 0; i < Nx; ++i) {
        double x = i * dx;
        for (int j = 0; j < Ny; ++j) {
            double y = j * dy;
            double u_exact = std::sin(2.0 * M_PI * x) * std::sin(2.0 * M_PI * y);
            acc += sqr(u[i][j] - u_exact);
        }
    }
    return std::sqrt(acc / (Nx * Ny));
}

// Pretty printing of params
void print_header(const Params& p) {
    std::cout << "==== Baseline Jacobi (serial) ====\n"
              << "Grid: " << p.Nx << " x " << p.Ny
              << "   tol: " << p.tol
              << "   max_iters: " << p.max_iters << "\n";
}

int main(int argc, char* argv[]) {
    Params P;
    parse_args(argc, argv, P);
    print_header(P);

    const double dx = P.Lx / (P.Nx - 1);
    const double dy = P.Ly / (P.Ny - 1);

    // Allocate & initialize
    Grid u     = make_grid(P.Nx, P.Ny, 0.0);
    Grid u_new = make_grid(P.Nx, P.Ny, 0.0);
    Grid rhs   = make_grid(P.Nx, P.Ny, 0.0);

    zero_boundary(u);
    zero_boundary(u_new);
    fill_rhs(rhs, P.Lx, P.Ly);

    // Iteration
    int iter = 0;
    double diff = 1.0;

    struct timeval t0, t1;
    gettimeofday(&t0, NULL);

    while (iter < P.max_iters && diff > P.tol) {
        diff = jacobi_sweep(u, u_new, rhs, dx, dy);

        // swap roles (u_new becomes current)
        std::swap(u, u_new);
        ++iter;

        if (P.print_every > 0 && (iter % P.print_every == 0)) {
            std::cout << "Iter " << std::setw(6) << iter
                      << "  ||u^{k+1}-u^{k}||_2 = " << std::scientific << diff << "\n";
        }
    }

    gettimeofday(&t1, NULL);
    double elapsed = (t1.tv_sec - t0.tv_sec);
    elapsed += (t1.tv_usec - t0.tv_usec) / 1000000.0;

    // Final error vs exact
    double err_exact = l2_error_vs_exact(u, P.Lx, P.Ly);

    // Report
    std::cout << "\n--------------------------------\n";
    std::cout << "Converged iters : " << iter << "\n";
    std::cout << "Last L2 diff    : " << std::scientific << diff << "\n";
    std::cout << "Elapsed (s)     : " << std::fixed << std::setprecision(6) << elapsed << "\n";
    std::cout << "L2 error vs exact: " << std::scientific << err_exact << "\n";
    std::cout << "--------------------------------\n";

    return 0;
}