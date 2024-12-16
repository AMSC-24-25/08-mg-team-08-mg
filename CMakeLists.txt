cmake_minimum_required(VERSION 3.10)
project(PoissonSolver LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
include_directories(include)

# Source files
set(SOURCES
    src/PoissonSolver.cpp
    src/PoissonSolverParallel.cpp  # Add this line
    src/main.cpp
)

# Define executable
add_executable(PoissonSolver ${SOURCES})