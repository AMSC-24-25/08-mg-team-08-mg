#include "PoissonSolver.h"
#include <iostream>
#include <iomanip>

int main() {
    int N = 11;              
    double a = 1.0;           
    int max_iter = 1000;      
    double tolerance = 1e-6;  

    PoissonSolver solver(N, a, max_iter, tolerance);

    // Test the initialize and analytical_solution methods
    solver.solve();

    return 0;
}