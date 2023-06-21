# Experiments with solving Least Squares Problems

This repo is a space for me to experiment with least squares problems.
It started with an initial implementation of the Levenberg-Marquardt algorithm as described in "Methods for Non-Linear Least Squares Problems" by K. Madsen/H. B. Nielsen/O. Tingleff (2004).

Note: If you're searching for a serious NLS sovler, then check out the excellent [Ceres Solver](http://ceres-solver.org/).

## Dependencies

- fmt
- spdlog
- Ceres Solver (optional, for comparison tests)

```
brew install spdlog fmt ceres-solver
```

## How to build

```
cd nls
cmake -DCMAKE_BUILD_TYPE=Release" -B build .
cmake --build build -j
```

## Run

```
./build/nls_test
./build/ceres_test
./build/robust_nls
```
