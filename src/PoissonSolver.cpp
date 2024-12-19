#include "../include/PoissonSolver.hpp"  
#include <cmath>            
#include <iostream>
#include <fstream>
#include <iomanip> 
using namespace std;

// Constructor
PoissonSolver::PoissonSolver(int N, double a, int max_iter, double tolerance, int levels, const std::string &boundary_path)
    : N(N), a(a), max_iter(max_iter), tolerance(tolerance), levels(levels), boundary_path(boundary_path) {
    u = vector<vector<double>>(N, vector<double>(N, 0.0));
    rhs = vector<vector<double>>(N, vector<double>(N, 0.0));
}

// Function example from docs
double PoissonSolver::analytical_solution(double x, double y) const {
    return exp(x) * exp(-2.0 * y);
}

double PoissonSolver::forcing_function(double x, double y) const {
    return -5.0 * exp(x) * exp(-2.0 * y);
}

void PoissonSolver::initialize() {
    double h = 1.0 / (N - 1);

    // If boundary_path is given, try to read boundary values from file
    bool use_file_boundary = !boundary_path.empty();
    std::vector<std::vector<double>> boundary_values;

    if (use_file_boundary) {
        std::ifstream boundary_file(boundary_path);
        if (!boundary_file) {
            std::cerr << "Warning: Cannot open boundary file: " << boundary_path << ". Using analytical boundary.\n";
            use_file_boundary = false;
        } else {
            // expect N lines with N values each
            boundary_values.resize(N, std::vector<double>(N, 0.0));
            for (int i = 0; i < N; ++i) {
                for (int j = 0; j < N; ++j) {
                    if (!(boundary_file >> boundary_values[i][j])) {
                        std::cerr << "Error reading boundary value at (" << i << "," << j << "). Using analytical boundary.\n";
                        use_file_boundary = false;
                        break;
                    }
                }
                if (!use_file_boundary) break;
            }
            boundary_file.close();
        }
    }


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

// Jacobi smoother
void PoissonSolver::jacobi_smooth(int num_sweeps) {
    double h2_a = (1.0 / (N - 1)) * (1.0 / (N - 1)) / a;

    // Temporary storage for updated values
    vector<vector<double>> u_new(N, vector<double>(N, 0.0));

    for (int sweep = 0; sweep < num_sweeps; ++sweep) {
        // Compute all updated values based on the old grid u
        for (int i = 1; i < N - 1; ++i) {
            for (int j = 1; j < N - 1; ++j) {
                u_new[i][j] = 0.25 * (u[i + 1][j] + u[i - 1][j] + u[i][j + 1] + u[i][j - 1] - h2_a * rhs[i][j]);
            }
        }

        // Copy u_new back into u for the next sweep
        // for (int i = 1; i < N - 1; ++i) {
        //     for (int j = 1; j < N - 1; ++j) {
        //         u[i][j] = u_new[i][j];
        //     }
        // }
        u.swap(u_new);
    }
}

vector<vector<double>> PoissonSolver::compute_residual() const {
    double h2_alpha = (1.0 / (N - 1)) * (1.0 / (N - 1)) / a;
    vector<vector<double>> residual(N,vector<double>(N, 0.0));
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
vector<vector<double>> PoissonSolver::restrict_residual(const vector<vector<double>> &fine_grid) const {
    int coarse_N = (N + 1) / 2;
    vector<vector<double>> coarse_grid(coarse_N, vector<double>(coarse_N, 0.0));

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
vector<vector<double>> PoissonSolver::prolong_correction(const vector<vector<double>> &coarse_grid) const {
    int fine_N = (coarse_grid.size() - 1) * 2 + 1;
    vector<vector<double>> fine_grid(fine_N, vector<double>(fine_N, 0.0));

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


void PoissonSolver::v_cycle(int level, std::vector<std::vector<double>> &u_level, 
                                      std::vector<std::vector<double>> &rhs_level, int num_levels) {
    // Pre-smoothing
    jacobi_smooth(5);

    if (level < num_levels - 1) {
        auto residual = compute_residual();

        // Restrict residual to coarser grid
        auto coarse_residual = restrict_residual(residual);

        int coarse_N = (u_level.size() + 1) / 2;
        vector<vector<double>> coarse_u(coarse_N, vector<double>(coarse_N, 0.0));

        // Recursive call to the V-cycle on the coarser grid
        v_cycle(level + 1, coarse_u, coarse_residual, num_levels);

        // Prolong correction to finer grid
        auto correction = prolong_correction(coarse_u);

        // Apply correction to the finer grid solution
        for (int i = 1; i < u_level.size() - 1; ++i) {
            for (int j = 1; j < u_level.size() - 1; ++j) {
                u_level[i][j] += correction[i][j];
            }
        }
    }

    // Post-smoothing
    jacobi_smooth(5);
}

// Solve using multigrid
vector<double> PoissonSolver::solve() {
    initialize(); // Initialize the grid and RHS
    int iter = 0;
    double error = 1e10;
    vector<double> errors; // Store errors for plotting

    while (iter < max_iter && error > tolerance) {
        // Perform one V-cycle
        v_cycle(0, u, rhs, levels);

        // Compute residual and error
        auto residual = compute_residual();
        error = 0.0;

        for (int i = 1; i < N - 1; ++i) {
            for (int j = 1; j < N - 1; ++j) {
                error += residual[i][j] * residual[i][j];
            }
        }
        error = sqrt(error);

        // Store the error
        errors.push_back(error);

        iter++;
    }

    return errors;
}


//solve using plain Iterative method
vector<double> PoissonSolver::solve_iterative() {
    initialize(); // Initialize the grid and RHS
    int iter = 0;
    double error = 1e10;
    vector<double> errors;

    while (iter < max_iter && error > tolerance) {
        // Perform one iterative sweep
        jacobi_smooth(1);

        // Compute residual and error
        auto residual = compute_residual();
        error = 0.0;

        for (int i = 1; i < N - 1; ++i) {
            for (int j = 1; j < N - 1; ++j) {
                error += residual[i][j] * residual[i][j];
            }
        }
        error = sqrt(error);

        // Store the error
        errors.push_back(error);

        iter++;
    }

    return errors;
}