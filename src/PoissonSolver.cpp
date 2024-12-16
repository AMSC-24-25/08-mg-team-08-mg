#include "PoissonSolver.h"  
#include <cmath>            
#include <iostream>        
#include <iomanip> 

/// Constructor
PoissonSolver::PoissonSolver(int N, double a, int max_iter, double tolerance)
    : N(N), a(a), max_iter(max_iter), tolerance(tolerance) {
    u = std::vector<std::vector<double>>(N, std::vector<double>(N, 0.0));
    rhs = std::vector<std::vector<double>>(N, std::vector<double>(N, 0.0));
}

// Function example from docs
double PoissonSolver::analytical_solution(double x, double y) {
    return std::exp(x) * std::exp(-2.0 * y);
}

double PoissonSolver::forcing_function(double x, double y) {
    return -5.0 * std::exp(x) * std::exp(-2.0 * y);
}

void PoissonSolver::initialize() {
    double h = 1.0 / (N - 1);
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
void PoissonSolver::gauss_seidel_smooth(int num_sweeps) {
    double h2_a = (1.0 / (N - 1)) * (1.0 / (N - 1)) / a;
    for (int sweep = 0; sweep < num_sweeps; ++sweep) {
        for (int i = 1; i < N - 1; ++i) {
            for (int j = 1; j < N - 1; ++j) {
                u[i][j] = 0.25 * (u[i + 1][j] + u[i - 1][j] + u[i][j + 1] + u[i][j - 1] - h2_a * rhs[i][j]);
            }
        }
    }
}

// Compute residual
std::vector<std::vector<double>> PoissonSolver::compute_residual() {
    double h2_alpha = (1.0 / (N - 1)) * (1.0 / (N - 1)) / a;
    std::vector<std::vector<double>> residual(N, std::vector<double>(N, 0.0));
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
std::vector<std::vector<double>> PoissonSolver::restrict_residual(const std::vector<std::vector<double>> &fine_grid) {
    int coarse_N = (N + 1) / 2;
    std::vector<std::vector<double>> coarse_grid(coarse_N, std::vector<double>(coarse_N, 0.0));
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

// Prolong correction to finer grid
std::vector<std::vector<double>> PoissonSolver::prolong_correction(const std::vector<std::vector<double>> &coarse_grid) {
    int fine_N = (N - 1) * 2 + 1;
    std::vector<std::vector<double>> fine_grid(fine_N, std::vector<double>(fine_N, 0.0));
    for (int i = 0; i < coarse_grid.size(); ++i) {
        for (int j = 0; j < coarse_grid[0].size(); ++j) {
            fine_grid[2 * i][2 * j] = coarse_grid[i][j];
        }
    }
    return fine_grid;
}

// Solve using multigrid
void PoissonSolver::solve() {
    initialize();
    int iter = 0;
    double error = 1e10;
    while (iter < max_iter && error > tolerance) {
        gauss_seidel_smooth(5);
        auto residual = compute_residual();
        error = 0.0;
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