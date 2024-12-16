#include "PoissonSolver.h"

int main() {
    int N = 21;                // Grid size
    double a = 1.0;            // Scaling constant
    int max_iter = 10000;      // Maximum iterations
    double tolerance = 1e-8;   // Convergence tolerance

    PoissonSolver solver(N, a, max_iter, tolerance);
    solver.solve();

    return 0;
}