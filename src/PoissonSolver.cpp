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

void PoissonSolver::solve() {
    std::cout << "Initializing the grid and right-hand side..." << std::endl;
    initialize();

    std::cout << "Testing initialized solution grid (u) and right-hand side (rhs):\n";

    std::cout << "\nSolution Grid (u):\n";
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << std::setw(10) << u[i][j] << " ";
        }
        std::cout << "\n";
    }

    std::cout << "\nRight-Hand Side (rhs):\n";
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            std::cout << std::setw(10) << rhs[i][j] << " ";
        }
        std::cout << "\n";
    }
}