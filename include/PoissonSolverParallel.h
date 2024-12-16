#ifndef POISSON_SOLVER_PARALLEL_H
#define POISSON_SOLVER_PARALLEL_H

#include <vector>

class PoissonSolverParallel {
public:
    // Constructor
    PoissonSolverParallel(int N, double a, int max_iter, double tolerance);

    // Analytical solution
    double analytical_solution(double x, double y);

    // Forcing function
    double forcing_function(double x, double y);

    // Initialize the solution grid and RHS
    void initialize();

    // Gauss-Seidel smoother
    void gauss_seidel_smooth(int num_sweeps);

    // Compute the residual
    std::vector<std::vector<double>> compute_residual();

    // Restriction to coarser grid
    std::vector<std::vector<double>> restrict_residual(const std::vector<std::vector<double>> &fine_grid);

    // Prolongation to finer grid
    std::vector<std::vector<double>> prolong_correction(const std::vector<std::vector<double>> &coarse_grid);

    // Perform V-cycle multigrid iteration
    void v_cycle(int level, std::vector<std::vector<double>> &u_level,
                 std::vector<std::vector<double>> &rhs_level, int num_levels);

    // Solve using multigrid
    void solve();

private:
    int N; // Grid size
    double a; // Coefficient
    int max_iter; // Maximum iterations
    double tolerance; // Convergence tolerance
    std::vector<std::vector<double>> u; // Solution grid
    std::vector<std::vector<double>> rhs; // Right-hand side grid
};

#endif // POISSON_SOLVER_PARALLEL_H