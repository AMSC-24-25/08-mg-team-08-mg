#include "../include/PoissonSolverParallel.hpp"
#include <cmath>
#include <iostream>
#include <iomanip>
#include <omp.h> // Include OpenMP header

/// Constructor
PoissonSolverParallel::PoissonSolverParallel(int N, double a, int max_iter, double tolerance, int levels)
    : N(N), a(a), max_iter(max_iter), tolerance(tolerance), levels(levels) {
    u = std::vector<std::vector<double>>(N, std::vector<double>(N, 0.0));
    rhs = std::vector<std::vector<double>>(N, std::vector<double>(N, 0.0));
}

// Function example from docs
double PoissonSolverParallel::analytical_solution(double x, double y) {
    return std::exp(x) * std::exp(-2.0 * y);
}

double PoissonSolverParallel::forcing_function(double x, double y) {
    return -5.0 * std::exp(x) * std::exp(-2.0 * y);
}

void PoissonSolverParallel::initialize() {
    double h = 1.0 / (N - 1);
    #pragma omp parallel for collapse(2) num_threads(8) // Parallelize with 8 threads
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
        #pragma omp parallel for collapse(2) num_threads(8) // Parallelize with 8 threads
        for (int i = 1; i < N - 1; ++i) {
            for (int j = 1; j < N - 1; ++j) {
                u[i][j] = 0.25 * (u[i + 1][j] + u[i - 1][j] + u[i][j + 1] + u[i][j - 1] - h2_a * rhs[i][j]);
            }
        }
    }
}

// Compute residual
std::vector<std::vector<double>> PoissonSolverParallel::compute_residual() {
    double h2_alpha = (1.0 / (N - 1)) * (1.0 / (N - 1)) / a;
    std::vector<std::vector<double>> residual(N, std::vector<double>(N, 0.0));
    
    #pragma omp parallel for collapse(2) num_threads(8) // Parallelize with 8 threads
    for (int i = 1; i < N - 1; ++i) {
        for (int j = 1; j < N - 1; ++j) {
            residual[i][j] = rhs[i][j] - (
                a * (u[i + 1][j] + u[i - 1][j] + u[i][j + 1] + u[i][j - 1] - 4 * u[i][j]) / h2_alpha
            );
        }
    }
    return residual;
}

// Restrict residual to coarser grid
std::vector<std::vector<double>> PoissonSolverParallel::restrict_residual(const std::vector<std::vector<double>> &fine_grid) {
    int coarse_N = (N + 1) / 2;
    std::vector<std::vector<double>> coarse_grid(coarse_N, std::vector<double>(coarse_N, 0.0));

    #pragma omp parallel for collapse(2) num_threads(8) // Parallelize with 8 threads
    for (int i = 1; i < coarse_N - 1; ++i) {
        for (int j = 1; j < coarse_N - 1; ++j) {
            double central = 0.25 * fine_grid[2 * i][2 * j];
            double cardinal = 0.125 * (
                fine_grid[2 * i - 1][2 * j] + 
                fine_grid[2 * i + 1][2 * j] + 
                fine_grid[2 * i][2 * j - 1] + 
                fine_grid[2 * i][2 * j + 1]
            );
            double diagonal = 0.0625 * (
                fine_grid[2 * i - 1][2 * j - 1] + 
                fine_grid[2 * i - 1][2 * j + 1] + 
                fine_grid[2 * i + 1][2 * j - 1] + 
                fine_grid[2 * i + 1][2 * j + 1]
            );
            coarse_grid[i][j] = central + cardinal + diagonal;
        }
    }

    return coarse_grid;
}

// Prolong correction to finer grid
std::vector<std::vector<double>> PoissonSolverParallel::prolong_correction(const std::vector<std::vector<double>> &coarse_grid) {
    int fine_N = (coarse_grid.size() - 1) * 2 + 1;
    std::vector<std::vector<double>> fine_grid(fine_N, std::vector<double>(fine_N, 0.0));

    #pragma omp parallel for collapse(2) num_threads(8) // Parallelize with 8 threads
    for (int i = 0; i < coarse_grid.size(); ++i) {
        for (int j = 0; j < coarse_grid[0].size(); ++j) {
            fine_grid[2 * i][2 * j] += coarse_grid[i][j];
            if (2 * i - 1 >= 0) fine_grid[2 * i - 1][2 * j] += 0.5 * coarse_grid[i][j];
            if (2 * i + 1 < fine_N) fine_grid[2 * i + 1][2 * j] += 0.5 * coarse_grid[i][j];
            if (2 * j - 1 >= 0) fine_grid[2 * i][2 * j - 1] += 0.5 * coarse_grid[i][j];
            if (2 * j + 1 < fine_N) fine_grid[2 * i][2 * j + 1] += 0.5 * coarse_grid[i][j];
            if (2 * i - 1 >= 0 && 2 * j - 1 >= 0) fine_grid[2 * i - 1][2 * j - 1] += 0.25 * coarse_grid[i][j];
            if (2 * i - 1 >= 0 && 2 * j + 1 < fine_N) fine_grid[2 * i - 1][2 * j + 1] += 0.25 * coarse_grid[i][j];
            if (2 * i + 1 < fine_N && 2 * j - 1 >= 0) fine_grid[2 * i + 1][2 * j - 1] += 0.25 * coarse_grid[i][j];
            if (2 * i + 1 < fine_N && 2 * j + 1 < fine_N) fine_grid[2 * i + 1][2 * j + 1] += 0.25 * coarse_grid[i][j];
        }
    }

    return fine_grid;
}

// V-cycle
void PoissonSolverParallel::v_cycle(int level, std::vector<std::vector<double>> &u_level, 
                                      std::vector<std::vector<double>> &rhs_level, int num_levels) {
    gauss_seidel_smooth(5);

    if (level < num_levels - 1) {
        auto residual = compute_residual();
        auto coarse_residual = restrict_residual(residual);
        int coarse_N = (u_level.size() + 1) / 2;
        std::vector<std::vector<double>> coarse_u(coarse_N, std::vector<double>(coarse_N, 0.0));
        v_cycle(level + 1, coarse_u, coarse_residual, num_levels);
        auto correction = prolong_correction(coarse_u);
        #pragma omp parallel for collapse(2) num_threads(8)
        for (int i = 1; i < u_level.size() - 1; ++i) {
            for (int j = 1; j < u_level.size() - 1; ++j) {
                u_level[i][j] += correction[i][j];
            }
        }
    }
    gauss_seidel_smooth(5);
}

// Solve using multigrid
std::vector<double> PoissonSolverParallel::solve() {
    initialize();
    int iter = 0;
    double error = 1e10;
    std::vector<double> errors;

    while (iter < max_iter && error > tolerance) {
        v_cycle(0, u, rhs, levels);
        auto residual = compute_residual();
        error = 0.0;
        #pragma omp parallel for reduction(+:error) collapse(2) num_threads(8)
        for (int i = 1; i < N - 1; ++i) {
            for (int j = 1; j < N - 1; ++j) {
                error += residual[i][j] * residual[i][j];
            }
        }
        error = std::sqrt(error);
        errors.push_back(error);
        iter++;
    }

    return errors;
}

// Solve using plain iterative method
std::vector<double> PoissonSolverParallel::solve_iterative() {
    initialize();
    int iter = 0;
    double error = 1e10;
    std::vector<double> errors;

    while (iter < max_iter && error > tolerance) {
        gauss_seidel_smooth(1);
        auto residual = compute_residual();
        error = 0.0;
        #pragma omp parallel for reduction(+:error) collapse(2) num_threads(8)
        for (int i = 1; i < N - 1; ++i) {
            for (int j = 1; j < N - 1; ++j) {
                error += residual[i][j] * residual[i][j];
            }
        }
        error = std::sqrt(error);
        errors.push_back(error);
        iter++;
    }

    return errors;
}