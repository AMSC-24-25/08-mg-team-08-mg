cmake_minimum_required(VERSION 3.28)
project(PoissonSolver LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
include_directories(include)
add_subdirectory(include/matplotplusplus)

# Source files
set(SOURCES
    src/PoissonSolver.cpp
    src/PoissonSolverParallel.cpp
    src/plot_errors.cpp
    src/main.cpp
)

# Define executable and link Matplot++ library
add_executable(PoissonSolver ${SOURCES})
target_link_libraries(PoissonSolver PUBLIC matplot)