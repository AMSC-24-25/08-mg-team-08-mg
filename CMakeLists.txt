cmake_minimum_required(VERSION 3.10)
project(PoissonSolver LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add source files
add_executable(PoissonSolver
    src/main.cpp
    src/PoissonSolver.cpp
)