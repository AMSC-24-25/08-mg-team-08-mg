#ifndef POISSON_SOLVER_H
#define POISSON_SOLVER_H

#include <vector>

class PoissonSolver {
public:
    /**
     * Constructor for PoissonSolver
     * @param N - Number of grid points in each dimension (N x N grid)
     * @param a - Scaling constant in the equation
     * @param max_iter - Maximum number of iterations
     * @param tolerance - Convergence tolerance
     */
    PoissonSolver(int N, double a, int max_iter, double tolerance);

    /**
     * Solve the Poisson equation
     */
    void solve();

private:
    int N;                     // Number of grid points
    double a;                  // Scaling constant
    int max_iter;              // Maximum number of iterations
    double tolerance;          // Convergence tolerance
    std::vector<std::vector<double>> u; // Solution grid
    std::vector<std::vector<double>> rhs; // Right-hand side (forcing term)

    /**
     * Initialize the solution grid and right-hand side
     * Applies Dirichlet boundary conditions
     */
    void initialize();

    /**
     * Analytical solution for testing purposes
     * @param x - x-coordinate
     * @param y - y-coordinate
     * @return Analytical solution value at (x, y)
     */
    double analytical_solution(double x, double y);

    /**
     * Forcing function for the Poisson equation
     * @param x - x-coordinate
     * @param y - y-coordinate
     * @return Forcing term value at (x, y)
     */
    double forcing_function(double x, double y);
};

/**
 * Perform Gauss-Seidel smoothing on the grid.
 * @param num_sweeps - Number of smoothing iterations
 */
void gauss_seidel_smooth(int num_sweeps);

#endif 