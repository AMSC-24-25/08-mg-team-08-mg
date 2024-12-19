#include "../include/PoissonSolverParallel.hpp"
#include <cmath>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <omp.h>


/// Constructor
PoissonSolverParallel::PoissonSolverParallel(int N, double a, int max_iter, double tolerance, int levels, int num_cores, const std::string &boundary_path)
    : N(N), a(a), max_iter(max_iter), tolerance(tolerance), levels(levels), num_cores(8), boundary_path(boundary_path) {
    u = std::vector<std::vector<double>>(N, std::vector<double>(N, 0.0));
    u_sol = std::vector<std::vector<double>>(N, std::vector<double>(N, 0.0));
    rhs = std::vector<std::vector<double>>(N, std::vector<double>(N, 0.0));
}

// Function example from docs
double PoissonSolverParallel::analytical_solution(double x, double y) const {
    return std::exp(x) * std::exp(-2.0 * y);
}

double PoissonSolverParallel::forcing_function(double x, double y) const {
    return -5.0 * std::exp(x) * std::exp(-2.0 * y);
}

void PoissonSolverParallel::initialize() {
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

    #pragma omp parallel num_threads(num_cores)
    {
        // Compute rhs and u_sol
        #pragma omp for collapse(2) schedule(static)
        for (int i = 0; i < N; ++i) {
            for (int j = 0; j < N; ++j) {
                double x = i * h;
                double y = j * h;
                rhs[i][j] = forcing_function(x, y);
                u_sol[i][j] = analytical_solution(x, y);
            }
        }

        // Top and bottom boundaries
        #pragma omp for schedule(static)
        for (int j = 0; j < N; ++j) {
            u[0][j]    = u_sol[0][j];
            u[N-1][j]  = u_sol[N-1][j];
        }

        // Left and right boundaries
        #pragma omp for schedule(static)
        for (int i = 0; i < N; ++i) {
            u[i][0]    = u_sol[i][0];
            u[i][N-1]  = u_sol[i][N-1];
        }
    }
}

// Jacobi smoother
void PoissonSolverParallel::jacobi_smooth(int num_sweeps) {
    double h2_a = (1.0 / (N - 1)) * (1.0 / (N - 1)) / a;
    std::vector<std::vector<double>> u_new(N, std::vector<double>(N, 0.0));

    for (int sweep = 0; sweep < num_sweeps; ++sweep) {
        #pragma omp parallel for collapse(2) schedule(static) num_threads(num_cores)
        for (int i = 1; i < N - 1; ++i) {
            for (int j = 1; j < N - 1; ++j) {
                u_new[i][j] = 0.25 * (u[i+1][j] + u[i-1][j] + u[i][j+1] + u[i][j-1] - h2_a * rhs[i][j]);
            }
        }

        #pragma omp parallel for collapse(2) schedule(static) num_threads(num_cores)
        for (int i = 1; i < N - 1; ++i) {
            for (int j = 1; j < N - 1; ++j) {
                u[i][j] = u_new[i][j];
            }
        }
    }
}

// Compute residual
std::vector<std::vector<double>> PoissonSolverParallel::compute_residual() const {
    double h2_alpha = (1.0/(N-1))*(1.0/(N-1))/a;
    std::vector<std::vector<double>> residual(N, std::vector<double>(N, 0.0));

    #pragma omp parallel for collapse(2) schedule(static) num_threads(num_cores)
    for (int i = 1; i < N - 1; ++i) {
        for (int j = 1; j < N - 1; ++j) {
            residual[i][j] = rhs[i][j] - (
                a * (u[i+1][j] + u[i-1][j] + u[i][j+1] + u[i][j-1] - 4 * u[i][j]) / h2_alpha
            );
        }
    }
    return residual;
}

// Restrict residual to coarser grid
std::vector<std::vector<double>> PoissonSolverParallel::restrict_residual(const std::vector<std::vector<double>> &fine_grid) const {
    int coarse_N = (N+1)/2;
    std::vector<std::vector<double>> coarse_grid(coarse_N, std::vector<double>(coarse_N,0.0));

    #pragma omp parallel for collapse(2) schedule(static) num_threads(num_cores)
    for (int i = 1; i < coarse_N - 1; ++i) {
        for (int j = 1; j < coarse_N - 1; ++j) {
            double central = 0.25 * fine_grid[2*i][2*j];
            double cardinal = 0.125 * (
                fine_grid[2*i-1][2*j] +
                fine_grid[2*i+1][2*j] +
                fine_grid[2*i][2*j-1] +
                fine_grid[2*i][2*j+1]
            );
            double diagonal = 0.0625 * (
                fine_grid[2*i-1][2*j-1] +
                fine_grid[2*i-1][2*j+1] +
                fine_grid[2*i+1][2*j-1] +
                fine_grid[2*i+1][2*j+1]
            );
            coarse_grid[i][j] = central + cardinal + diagonal;
        }
    }

    return coarse_grid;
}

// Prolong correction to finer grid
std::vector<std::vector<double>> PoissonSolverParallel::prolong_correction(const std::vector<std::vector<double>> &coarse_grid) const {
    int fine_N = (int(coarse_grid.size()) - 1) * 2 + 1;
    std::vector<std::vector<double>> padded_grid(fine_N+2, std::vector<double>(fine_N+2,0.0));

    int coarse_size = (int)coarse_grid.size();
    int coarse_size_j = (int)coarse_grid[0].size();
    #pragma omp parallel for collapse(2) schedule(static) num_threads(num_cores)
    for (int i = 0; i < coarse_size; ++i) {
        for (int j = 0; j < coarse_size_j; ++j) {
            double val = coarse_grid[i][j];
            int I = 2*i + 1;
            int J = 2*j + 1;

            padded_grid[I][J] += val;
            padded_grid[I-1][J] += 0.5*val;
            padded_grid[I+1][J] += 0.5*val;
            padded_grid[I][J-1] += 0.5*val;
            padded_grid[I][J+1] += 0.5*val;

            padded_grid[I-1][J-1] += 0.25*val;
            padded_grid[I-1][J+1] += 0.25*val;
            padded_grid[I+1][J-1] += 0.25*val;
            padded_grid[I+1][J+1] += 0.25*val;
        }
    }

    std::vector<std::vector<double>> fine_grid(fine_N, std::vector<double>(fine_N,0.0));
    #pragma omp parallel for collapse(2) schedule(static) num_threads(num_cores)
    for (int i = 0; i < fine_N; ++i) {
        for (int j = 0; j < fine_N; ++j) {
            fine_grid[i][j] = padded_grid[i+1][j+1];
        }
    }

    return fine_grid;
}

// V-cycle
void PoissonSolverParallel::v_cycle(int level, std::vector<std::vector<double>> &u_level, 
                                      std::vector<std::vector<double>> &rhs_level, int num_levels) {
    jacobi_smooth(5);

    if (level < num_levels - 1) {
        auto residual = compute_residual();
        auto coarse_residual = restrict_residual(residual);
        int coarse_N = (u_level.size() + 1) / 2;
        std::vector<std::vector<double>> coarse_u(coarse_N, std::vector<double>(coarse_N, 0.0));
        v_cycle(level + 1, coarse_u, coarse_residual, num_levels);
        auto correction = prolong_correction(coarse_u);

        #pragma omp parallel for schedule(static) num_threads(num_cores)
        for (int i = 1; i < u_level.size() - 1; ++i) {
            for (int j = 1; j < u_level.size() - 1; ++j) {
                u_level[i][j] += correction[i][j];
            }
        }
    }
    jacobi_smooth(5);
}

// Solve using multigrid
std::vector<double> PoissonSolverParallel::solve() {
    initialize();
    int iter = 0;
    double error = 1e10;
    std::vector<double> errors;

    while (iter < max_iter && error > tolerance) {
        v_cycle(0, u, rhs, levels);
        auto residual = compute_residual();
        error = 0.0;
        #pragma omp parallel for reduction(+:error) collapse(2) num_threads(num_cores)
        for (int i = 1; i < N - 1; ++i) {
            for (int j = 1; j < N - 1; ++j) {
                error += residual[i][j] * residual[i][j];
            }
        }
        error = std::sqrt(error);
        errors.push_back(error);
        iter++;
    }

    return errors;
}

// Solve using plain iterative method
std::vector<double> PoissonSolverParallel::solve_iterative() {
    initialize();
    int iter = 0;
    double error = 1e10;
    std::vector<double> errors;

    while (iter < max_iter && error > tolerance) {
        jacobi_smooth(1);
        auto residual = compute_residual();
        error = 0.0;
        #pragma omp parallel for reduction(+:error) collapse(2) num_threads(num_cores)
        for (int i = 1; i < N - 1; ++i) {
            for (int j = 1; j < N - 1; ++j) {
                error += residual[i][j] * residual[i][j];
            }
        }
        error = std::sqrt(error);
        errors.push_back(error);
        iter++;
    }

    return errors;
}

double PoissonSolverParallel::determine_error() {
    double error_L2 = 0.0;

    // Compute errors in parallel
    #pragma omp parallel for reduction(+:error_L2) collapse(2) num_threads(num_cores)
    for (int i = 0; i < N - 1; ++i) {
        for (int j = 0; j < N - 1; ++j) {
            double diff = u_sol[i][j] - u[i][j];
            error_L2 += diff * diff;
        }
    }

    // Finalize norms
    double L2_norm = std::sqrt(error_L2);

    // Return one of the metrics if the function needs to return a single value
    return L2_norm; // For example, returning the L2 norm
}


