#include "PoissonSolver.hpp"
#include "PoissonSolverParallel.hpp"
#include <iostream>

int main() {
    int N = 21;                // Grid size
    double a = 1.0;            // Scaling constant
    int max_iter = 10000;      // Maximum iterations
    double tolerance = 1e-8;   // Convergence tolerance

    // Run the serial PoissonSolver
    PoissonSolver solver(N, a, max_iter, tolerance);
    std::cout << "Running Serial PoissonSolver...\n";
    solver.solve();

    // Run the parallel PoissonSolverParallel
    PoissonSolverParallel parallelSolver(N, a, max_iter, tolerance);
    std::cout << "Running Parallel PoissonSolverParallel...\n";
    parallelSolver.solve();

    return 0;
}