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

// Function to plot errors for multiple levels
void plot_errors(std::vector<int> levels) {
    auto fig = figure(true);
    hold(on); // Allow multiple plots on the same figure

    // Plot "No Multigrid" data
    std::vector<int> plain_iterations;
    std::vector<double> plain_errors;
    read_csv("plain_errors.csv", plain_iterations, plain_errors);
    std::vector<double> plain_errors_log;
    for (auto &e : plain_errors) plain_errors_log.push_back(std::log10(e));
    plot(plain_iterations, plain_errors_log)->line_style("--").line_width(2).display_name("No MG");

    for (int level : levels) {
        // File name for the current level
        std::string filename = "multigrid_errors_level_" + std::to_string(level) + ".csv";

        // Read errors from file
        std::vector<int> iterations;
        std::vector<double> errors;
        read_csv(filename, iterations, errors);

        // Apply log10 transformation to errors
        std::vector<double> log_errors;
        for (auto &e : errors) log_errors.push_back(std::log10(e));

        // Plot the data
        plot(iterations, log_errors)->line_width(2).display_name("Levels = " + std::to_string(level));
    }

    hold(off);

    // Configure the plot
    xlabel("Iteration");
    ylabel("Log10(Error)");
    title("Error vs. Iteration for Multigrid Solver at Different Levels");
    legend();
    save("/home/jellyfish/shared-folder/AMSC/08-mg-team-08-mg/data/multigrid_convergence.png");
    std::cout << "Plot saved as 'multigrid_convergence.png'\n";
}