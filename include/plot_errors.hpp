#ifndef ERROR_PLOTTER_HPP
#define ERROR_PLOTTER_HPP

#include <matplot/matplot.h>
#include <vector>
#include <string>

// Function to read CSV data into vectors
void read_csv(const std::string &filename, std::vector<int> &iterations, std::vector<double> &errors);

// Function to plot the errors
void plot_errors(std::vector<int> levels);

#endif