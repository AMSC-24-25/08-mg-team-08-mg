#include "PoissonSolver.hpp"
#include "plot_errors.hpp"
#include <iostream>
#include <fstream>
#include <vector>

// Function to save errors to CSV
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
    int max_iter = 5000;       // Maximum iterations
    double tolerance = 1e-10;   // Convergence tolerance
    std::vector<int> levels = {1, 3, 5, 7};

    // Run the plain Gauss-Seidel solver
    std::cout << "Running Plain Gauss-Seidel Solver (No MG)...\n";
    PoissonSolver plainSolver(N, a, max_iter, tolerance, 1);
    std::vector<double> plain_errors = plainSolver.solve_iterative();
    save_errors("plain_errors.csv", plain_errors);

    // Run multigrid solver for different levels
    for (int level : levels) {
        std::cout << "Running Multigrid PoissonSolver with levels = " << level << "...\n";
        PoissonSolver solver(N, a, max_iter, tolerance, level);
        std::vector<double> errors = solver.solve();

        // Save errors to a file named "multigrid_errors_level_X.csv"
        std::string filename = "multigrid_errors_level_" + std::to_string(level) + ".csv";
        save_errors(filename, errors);
    }

    // Plot all results
    plot_errors(levels);

    return 0;
}