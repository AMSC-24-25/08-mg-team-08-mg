#include "PoissonSolverParallel.hpp"
#include <cmath>
#include <iostream>
#include <iomanip>
#include <omp.h>  // Include OpenMP header

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
        double x = i * h;
        for (int j = 0; j < N; ++j) {
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
            // Central coincident node
            double central = 0.25 * fine_grid[2 * i][2 * j];
            
            // North, South, East, West neighbors (cardinal directions)
            double cardinal = 0.125 * (
                fine_grid[2 * i - 1][2 * j] + // North
                fine_grid[2 * i + 1][2 * j] + // South
                fine_grid[2 * i][2 * j - 1] + // West
                fine_grid[2 * i][2 * j + 1]   // East
            );

            // Diagonal neighbors (North-East, North-West, South-East, South-West)
            double diagonal = 0.0625 * (
                fine_grid[2 * i - 1][2 * j - 1] + // North-West
                fine_grid[2 * i - 1][2 * j + 1] + // North-East
                fine_grid[2 * i + 1][2 * j - 1] + // South-West
                fine_grid[2 * i + 1][2 * j + 1]   // South-East
            );

            // Combine contributions
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
            // Transfer the value of the coincident coarse node
            fine_grid[2 * i][2 * j] += coarse_grid[i][j];

            // North, South, East, West neighbors (1/2 of the value)
            if (2 * i - 1 >= 0) fine_grid[2 * i - 1][2 * j] += 0.5 * coarse_grid[i][j]; // North
            if (2 * i + 1 < fine_N) fine_grid[2 * i + 1][2 * j] += 0.5 * coarse_grid[i][j]; // South
            if (2 * j - 1 >= 0) fine_grid[2 * i][2 * j - 1] += 0.5 * coarse_grid[i][j]; // West
            if (2 * j + 1 < fine_N) fine_grid[2 * i][2 * j + 1] += 0.5 * coarse_grid[i][j]; // East

            // Diagonal neighbors (1/4 of the value)
            if (2 * i - 1 >= 0 && 2 * j - 1 >= 0) fine_grid[2 * i - 1][2 * j - 1] += 0.25 * coarse_grid[i][j]; // North-West
            if (2 * i - 1 >= 0 && 2 * j + 1 < fine_N) fine_grid[2 * i - 1][2 * j + 1] += 0.25 * coarse_grid[i][j]; // North-East
            if (2 * i + 1 < fine_N && 2 * j - 1 >= 0) fine_grid[2 * i + 1][2 * j - 1] += 0.25 * coarse_grid[i][j]; // South-West
            if (2 * i + 1 < fine_N && 2 * j + 1 < fine_N) fine_grid[2 * i + 1][2 * j + 1] += 0.25 * coarse_grid[i][j]; // South-East
        }
    }

    return fine_grid;
}


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

        // Recursive call to the V-cycle on the coarser grid
        v_cycle(level + 1, coarse_u, coarse_residual, num_levels);

        // Prolong correction to finer grid
        auto correction = prolong_correction(coarse_u);

        // Apply correction to the finer grid solution
        #pragma omp parallel for collapse(2) num_threads(8) // Parallelize with 8 threads
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
std::vector<double> PoissonSolverParallel::solve() {
    initialize(); // Initialize the grid and RHS
    int iter = 0;
    double error = 1e10;
    std::vector<double> errors; // Store errors for plotting

    while (iter < max_iter && error > tolerance) {
        // Perform one V-cycle
        v_cycle(0, u, rhs, levels);

        // Compute residual and error
        auto residual = compute_residual();
        error = 0.0;

        #pragma omp parallel for reduction(+:error) collapse(2) num_threads(8) // Parallelize and reduce error computation
        for (int i = 1; i < N - 1; ++i) {
            for (int j = 1; j < N - 1; ++j) {
                error += residual[i][j] * residual[i][j];
            }
        }
        error = std::sqrt(error);

        // Store the error
        errors.push_back(error);

        iter++;
    }

    return errors;
}

//solve using plain Iterative method
std::vector<double> PoissonSolverParallel::solve_iterative() {
    initialize(); // Initialize the grid and RHS
    int iter = 0;
    double error = 1e10;
    std::vector<double> errors;

    while (iter < max_iter && error > tolerance) {
        // Perform one full Gauss-Seidel sweep
        gauss_seidel_smooth(1);

        // Compute residual and error
        auto residual = compute_residual();
        error = 0.0;

        #pragma omp parallel for reduction(+:error) collapse(2) num_threads(8) // Parallelize and reduce error computation
        for (int i = 1; i < N - 1; ++i) {
            for (int j = 1; j < N - 1; ++j) {
                error += residual[i][j] * residual[i][j];
            }
        }
        error = std::sqrt(error);

        // Store the error
        errors.push_back(error);

        iter++;
    }

    return errors;
}