#include <iostream>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <vector>
#include "PoissonSolver.hpp"
#include "plot_errors.hpp"

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

std::unordered_map<std::string, std::string> read_config(const std::string &filename) {
    std::unordered_map<std::string, std::string> config;
    std::ifstream file(filename);
    if (file.is_open()) {
        std::string line;
        while (std::getline(file, line)) {
            std::istringstream iss(line);
            std::string key, value;
            if (std::getline(iss, key, '=') && std::getline(iss, value)) {
                // Special handling for levels (comma-separated values)
                if (key == "levels") {
                    config[key] = value; // Store the string "1,3,5,7"
                } else {
                    config[key] = value; // Store other key-value pairs
                }
            }
        }
        file.close();
    } else {
        std::cerr << "Failed to open config file: " << filename << "\n";
    }
    return config;
}

// Helper function to convert the comma-separated string to a vector of integers
std::vector<int> parse_levels(const std::string &levels_str) {
    std::vector<int> levels;
    std::istringstream ss(levels_str);
    std::string temp;
    while (std::getline(ss, temp, ',')) {
        levels.push_back(std::stoi(temp)); // Convert each part to an integer and add to vector
    }
    return levels;
}

int main() {
    // Read the configuration from the config.txt file
    auto config = read_config("../data/config.txt");

    // Read values from the config file and convert them to appropriate types
    int N = std::stoi(config["N"]);
    double a = std::stod(config["a"]);
    int max_iter = std::stoi(config["max_iter"]);
    double tolerance = std::stod(config["tolerance"]);

    // Read the levels as a comma-separated string, then parse it into a vector of integers
    std::vector<int> levels = parse_levels(config["levels"]);

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