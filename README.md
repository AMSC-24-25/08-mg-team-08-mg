# Multigrid
This project solves the 2D Poisson equation:

$$
\begin{cases}
-\nabla \cdot a \nabla u = f & \text{in } \Omega, \\
u = g & \text{on } \partial \Omega.
\end{cases}
$$


### gauss_seidel_smooth method implementation
the number of smoothing iterations each sweep reduces high-frequency errors in the solution
Assuming that $a$ is a known constant, a uniform Cartesian grid with spacing $h$, the Laplacian $\nabla^2 u$ is approximated at a grid point $(i, j)$ as:

$$
\nabla^2 u \approx \frac{u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1} - 4u_{i,j}}{h^2}.
$$

Substituting this into the Poisson equation $-a \nabla^2 u = f$, we get:

$$
-a \cdot \frac{u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1} - 4u_{i,j}}{h^2} = f_{i,j}.
$$

Rearranging for $u_{i,j}$, we obtain the iterative update formula:

$$
u_{i,j} = \frac{1}{4} \left(u_{i+1,j} + u_{i-1,j} + u_{i,j+1} + u_{i,j-1} - \frac{h^2}{a} f_{i,j}\right).
$$



## How to use: 
### install dependency
```
apt-get install gnuplot
cd include
git clone https://github.com/alandefreitas/matplotplusplus.git
cd ..
```


### compile
```
mkdir -p build
cd build
cmake ..
make
```

### run
```
./PoissonSolver
cd ..
```