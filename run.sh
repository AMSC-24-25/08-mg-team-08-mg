#!/bin/bash

# Check if the build folder exists
if [ -d "build" ]; then
    echo "Build folder exists. Removing it..."
    rm -rf build
else
    echo "Build folder does not exist. Proceeding to create it..."
fi

# Create and navigate to the build folder, then build the project
mkdir -p build
cd build
cmake ..
make 
./PoissonSolver
cd ..