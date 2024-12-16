#include "PoissonSolver.hpp"
#include "plot_errors.hpp"
#include <iostream>
#include <fstream>


void save_errors(const std::string &filename, const std::vector<double> &errors) {
    std::ofstream file(filename);
    if (file.is_open()) {
        for (size_t i = 0; i < errors.size(); ++i) {
            file << i << "," << errors[i] << "\n"; // Save iteration and error
        }
        file.close();
    } else {
        std::cerr << "Failed to open file: " << filename << "\n";
    }
}

int main() {
    int N = 65;                 // Grid size
    double a = 1.0;             // Scaling constant
    int max_iter = 10000;       // Maximum iterations
    double tolerance = 1e-10;   // Convergence tolerance
    int levels = 3;             // Number of coarsening levels

    // Run the plain Gauss-Seidel solver
    PoissonSolver plainSolver(N, a, max_iter, tolerance, levels);
    std::vector<double> iterative_errors = plainSolver.solve_iterative();
    save_errors("iterative_errors.csv", iterative_errors);

    // Run the multigrid PoissonSolver
    PoissonSolver multigridSolver(N, a, max_iter, tolerance, levels);
    std::vector<double> multigrid_errors = multigridSolver.solve();
    save_errors("multigrid_errors.csv", multigrid_errors);
    plot_errors();

    return 0;
}