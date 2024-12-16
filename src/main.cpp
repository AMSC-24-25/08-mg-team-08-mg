#include "PoissonSolver.hpp"
#include "PoissonSolverParallel.hpp"
#include <iostream>

int main() {
    int N = 65;                // Grid size
    double a = 1.0;             // Scaling constant
    int max_iter = 10000;       // Maximum iterations
    double tolerance = 1e-10;   // Convergence tolerance

    // Run the plain Gauss-Seidel solver
    PoissonSolver plainSolver(N, a, max_iter, tolerance);
    std::cout << "Running Plain Gauss-Seidel Solver...\n";
    plainSolver.solve_plain_gauss_seidel();

    // Run the multigrid PoissonSolver
    PoissonSolver multigridSolver(N, a, max_iter, tolerance);
    std::cout << "Running Multigrid PoissonSolver...\n";
    multigridSolver.solve();

    return 0;
}