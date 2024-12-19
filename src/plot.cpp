#include "../include/plot.hpp"
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
    save("../data/multigrid_convergence.png");
    std::cout << "Plot saved as 'multigrid_convergence.png'\n";
}



void plot_times(const std::vector<double> &seq_duration, const std::vector<double> &par_duration, const std::vector<int> &levels) {
    // Convert levels (int) to double for plotting
    std::vector<double> double_levels(levels.begin(), levels.end());

    // Convert levels to string labels
    std::vector<std::string> labels;
    for (auto lvl : levels) {
        labels.push_back(std::to_string(lvl));
    }

    figure(true);
    auto ax = gca();

    // Plot the sequential bars
    auto b1 = bar(double_levels, seq_duration);
    b1->bar_width(0.4);               // Make the bars narrower
    b1->display_name("Sequential");   // Label these bars as Sequential

    hold(on);

    // We need to plot the parallel bars shifted a bit to the right
    // Create a shifted x-axis for parallel bars
    std::vector<double> shifted_levels = double_levels;
    for (auto &x : shifted_levels) {
        x += 0.4; // Shift by the same amount as the width of the first bars
    }

    auto b2 = bar(shifted_levels, par_duration);
    b2->bar_width(0.4);
    b2->display_name("Parallel");     // Label these bars as Parallel

    // Set the x-axis ticks at the original level positions
    ax->x_axis().tick_values(double_levels);
    ax->x_axis().ticklabels(labels);

    xlabel("Levels");
    ylabel("Time Duration (units)");
    title("Sequential vs Parallel Time Comparison");

    // Show legend
    legend();

    // Save the figure as PNG (replace with a valid path)
    save("../data/time_comparison.png");
    std::cout << "Plot saved as 'time_comparison.png'\n";
}