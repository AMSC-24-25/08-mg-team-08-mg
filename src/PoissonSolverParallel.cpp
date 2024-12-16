#include "PoissonSolverParallel.h"
#include <cmath>
#include <iostream>
#include <iomanip>
#include <omp.h>

// Constructor
PoissonSolverParallel::PoissonSolverParallel(int N, double a, int max_iter, double tolerance)
    : N(N), a(a), max_iter(max_iter), tolerance(tolerance) {
    u = std::vector<std::vector<double>>(N, std::vector<double>(N, 0.0));
    rhs = std::vector<std::vector<double>>(N, std::vector<double>(N, 0.0));
}

// Analytical solution
double PoissonSolverParallel::analytical_solution(double x, double y) {
    return std::exp(x) * std::exp(-2.0 * y);
}

// Forcing function
double PoissonSolverParallel::forcing_function(double x, double y) {
    return -5.0 * std::exp(x) * std::exp(-2.0 * y);
}

// Initialize the solution grid and RHS
void PoissonSolverParallel::initialize() {
    double h = 1.0 / (N - 1);
    #pragma omp parallel for collapse(2) // Parallelize nested loops
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            double x = i * h;
            double y = j * h;
            rhs[i][j] = forcing_function(x, y);
            if (i == 0 || i == N - 1 || j == 0 || j == N - 1) {
                u[i][j] = analytical_solution(x, y);
            }
        }
    }
}

// Gauss-Seidel smoother
void PoissonSolverParallel::gauss_seidel_smooth(int num_sweeps) {
    double h2_a = (1.0 / (N - 1)) * (1.0 / (N - 1)) / a;
    for (int sweep = 0; sweep < num_sweeps; ++sweep) {
        #pragma omp parallel for collapse(2) // Parallelize over interior points
        for (int i = 1; i < N - 1; ++i) {
            for (int j = 1; j < N - 1; ++j) {
                u[i][j] = 0.25 * (u[i + 1][j] + u[i - 1][j] + u[i][j + 1] + u[i][j - 1] - h2_a * rhs[i][j]);
            }
        }
    }
}

// Compute the residual
std::vector<std::vector<double>> PoissonSolverParallel::compute_residual() {
    double h2_alpha = (1.0 / (N - 1)) * (1.0 / (N - 1)) / a;
    std::vector<std::vector<double>> residual(N, std::vector<double>(N, 0.0));
    #pragma omp parallel for collapse(2) // Parallelize nested loops
    for (int i = 1; i < N - 1; ++i) {
        for (int j = 1; j < N - 1; ++j) {
            residual[i][j] = rhs[i][j] - (
                a * (u[i + 1][j] + u[i - 1][j] + u[i][j + 1] + u[i][j - 1] - 4 * u[i][j]) / h2_alpha
            );
        }
    }
    return residual;
}

// Restriction to coarser grid
std::vector<std::vector<double>> PoissonSolverParallel::restrict_residual(const std::vector<std::vector<double>> &fine_grid) {
    int coarse_N = (N + 1) / 2;
    std::vector<std::vector<double>> coarse_grid(coarse_N, std::vector<double>(coarse_N, 0.0));
    #pragma omp parallel for collapse(2) // Parallelize restriction
    for (int i = 1; i < coarse_N - 1; ++i) {
        for (int j = 1; j < coarse_N - 1; ++j) {
            coarse_grid[i][j] = 0.25 * (
                fine_grid[2 * i][2 * j] +
                0.5 * (fine_grid[2 * i - 1][2 * j] + fine_grid[2 * i + 1][2 * j]) +
                0.5 * (fine_grid[2 * i][2 * j - 1] + fine_grid[2 * i][2 * j + 1])
            );
        }
    }
    return coarse_grid;
}

// Prolongation to finer grid
std::vector<std::vector<double>> PoissonSolverParallel::prolong_correction(const std::vector<std::vector<double>> &coarse_grid) {
    int fine_N = (N - 1) * 2 + 1;
    std::vector<std::vector<double>> fine_grid(fine_N, std::vector<double>(fine_N, 0.0));
    #pragma omp parallel for collapse(2) // Parallelize prolongation
    for (int i = 0; i < coarse_grid.size(); ++i) {
        for (int j = 0; j < coarse_grid[0].size(); ++j) {
            fine_grid[2 * i][2 * j] = coarse_grid[i][j];
        }
    }
    return fine_grid;
}

// Perform V-cycle multigrid iteration
void PoissonSolverParallel::v_cycle(int level, std::vector<std::vector<double>> &u_level,
                                     std::vector<std::vector<double>> &rhs_level, int num_levels) {
    // Pre-smoothing
    gauss_seidel_smooth(5);

    if (level < num_levels - 1) {
        auto residual = compute_residual();

        // Restrict residual to coarser grid
        auto coarse_residual = restrict_residual(residual);

        int coarse_N = (u_level.size() + 1) / 2;
        std::vector<std::vector<double>> coarse_u(coarse_N, std::vector<double>(coarse_N, 0.0));

        // Recursive V-cycle on coarser grid
        v_cycle(level + 1, coarse_u, coarse_residual, num_levels);

        // Prolong correction to finer grid
        auto correction = prolong_correction(coarse_u);

        // Apply correction
        #pragma omp parallel for collapse(2) // Parallelize correction application
        for (int i = 1; i < u_level.size() - 1; ++i) {
            for (int j = 1; j < u_level.size() - 1; ++j) {
                u_level[i][j] += correction[i][j];
            }
        }
    }

    // Post-smoothing
    gauss_seidel_smooth(5);
}

// Solve using multigrid
void PoissonSolverParallel::solve() {
    initialize();
    int iter = 0;
    double error = 1e10;

    while (iter < max_iter && error > tolerance) {
        // Perform one V-cycle
        v_cycle(0, u, rhs, std::log2(N - 1));

        // Compute residual and error
        auto residual = compute_residual();
        error = 0.0;

        #pragma omp parallel for collapse(2) reduction(+:error) // Parallelize error computation
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                error += residual[i][j] * residual[i][j];
            }
        }

        error = std::sqrt(error);
        std::cout << "Iteration: " << iter << ", Error: " << error << "\n";
        iter++;
    }
}