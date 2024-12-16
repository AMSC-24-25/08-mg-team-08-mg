#include "plot_errors.hpp"
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <matplot/matplot.h>

using namespace matplot;

// Function to read CSV data into vectors
void read_csv(const std::string &filename, std::vector<int> &iterations, std::vector<double> &errors) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file: " + filename);
    }

    std::string line;
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string iter_str, error_str;
        if (std::getline(ss, iter_str, ',') && std::getline(ss, error_str, ',')) {
            iterations.push_back(std::stoi(iter_str));
            errors.push_back(std::stod(error_str));
        }
    }
    file.close();
}

// Function to plot the errors
void plot_errors() {
    // Data containers
    std::vector<int> plain_iterations, multigrid_iterations;
    std::vector<double> plain_errors, multigrid_errors;

    // Read CSV files
    read_csv("iterative_errors.csv", plain_iterations, plain_errors);
    read_csv("multigrid_errors.csv", multigrid_iterations, multigrid_errors);

    // Apply log10 transformation manually to errors for plotting
    std::vector<double> plain_errors_log, multigrid_errors_log;
    for (auto &e : plain_errors) plain_errors_log.push_back(std::log10(e));
    for (auto &e : multigrid_errors) multigrid_errors_log.push_back(std::log10(e));

    // Create the plot
    auto fig = figure(true);

    hold(on); // Keep multiple plots on the same figure

    // Plot data
    auto p1 = plot(plain_iterations, plain_errors_log);
    p1->line_width(2).line_style("--");
    p1->display_name("No MG");

    auto p2 = plot(multigrid_iterations, multigrid_errors_log);
    p2->line_width(2).line_style("-");
    p2->display_name("Multigrid");

    hold(off); // Release hold to finalize plotting

    // Configure axes
    xlabel("Iteration");
    ylabel("Log10(Error)");
    title("Error vs. Iteration: Gauss-Seidel vs. Multigrid (Log Scale)");
    legend();

    // Save the plot
    save("error_comparison.png");
    std::cout << "Plot saved in /build as 'error_comparison.png'" << std::endl;
}