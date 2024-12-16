int main() {
    int N = 21;            // Grid size
    int max_iter = 10000;
    double tolerance = 1e-6;
    double alpha = 1.0;

    std::cout << "Solving 2D Poisson Equation on " << N << "x" << N << " grid with alpha = " << alpha << ".\n";
    solve_poisson(N, max_iter, tolerance, alpha);

    return 0;
}