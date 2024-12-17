cmake_minimum_required(VERSION 3.28)
project(PoissonSolver LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories
include_directories(include)
add_subdirectory(include/matplotplusplus)

# Find OpenMP
find_package(OpenMP REQUIRED)

# Source files
set(SOURCES
    lib/PoissonSolver.cpp
    lib/PoissonSolverParallel.cpp
    lib/plot_errors.cpp
    src/main.cpp
)

# Define executable and link libraries
add_executable(PoissonSolver ${SOURCES})

# Link Matplot++ library
target_link_libraries(PoissonSolver PUBLIC matplot)

# Link OpenMP library to the target
target_link_libraries(PoissonSolver PUBLIC OpenMP::OpenMP_CXX)

# Optional: Print a message if OpenMP is found
if(OpenMP_CXX_FOUND)
    message(STATUS "OpenMP found and enabled")
endif()