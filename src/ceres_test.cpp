#include <ceres/ceres.h>
#include <glog/logging.h>
#include <spdlog/spdlog.h>

#include "eigen_fmt.h"

// generated with
// import numpy as np
// r = 1.0
// p = [1, 2, -3]
// xs, ys, zs = [], [], []
// for phi in np.linspace(-1.0, 1.0, 100):
//     for psi in np.linspace(-0.7, 0.7, 100):
//         n = 1e-2 * np.random.randn(3)
//         x = p[0] + r * np.cos(psi) * np.cos(phi) + n[0]
//         y = p[1] + r * np.cos(psi) * np.sin(phi) + n[1]
//         z = p[2] + r * np.sin(psi) + n[2]
//         print(f"{x},{y},{z},")

// clang-format off
const double data[] = {
#include "data.csv"
  // outliers...
  10, 20, 30,
};
const int numObservations = sizeof(data) / (3*sizeof(double));
// clang-format on

using ceres::AutoDiffCostFunction;
using ceres::CauchyLoss;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;

struct SphereResidual {
    SphereResidual(double x, double y, double z)
        : x_(x)
        , y_(y)
        , z_(z)
    {
    }
    template <typename T>
    bool operator()(const T* const p, const T* const radius, T* residual) const
    {
        residual[0] = radius[0] - sqrt(pow(p[0] - x_, 2) + pow(p[1] - y_, 2) + pow(p[2] - z_, 2));
        return true;
    }

private:
    const double x_;
    const double y_;
    const double z_;
};

int main(int argc, char** argv)
{
    google::InitGoogleLogging(argv[0]);
    double p[3] = {0.0, 0.0, 0.0};
    double r = 0.1;
    Problem problem;

    spdlog::info("Num Observations: {}", numObservations);
    for (int i = 0; i < numObservations; ++i) {
        const double* obs = &data[3 * i];
        CostFunction* cost_function =
            new AutoDiffCostFunction<SphereResidual, 1, 3, 1>(
                new SphereResidual(obs[0], obs[1], obs[2]));
        problem.AddResidualBlock(cost_function, new CauchyLoss(0.5), p, &r);
        // problem.AddResidualBlock(cost_function, 0, p, &r);
    }
    Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.minimizer_progress_to_stdout = true;
    Solver::Summary summary;

    spdlog::info("solve");
    Solve(options, &problem, &summary);

    std::cout << summary.BriefReport() << "\n";
    spdlog::info("p: [{}, {}, {}]", p[0], p[1], p[2]);
    spdlog::info("r: {}", r);
    return 0;
}
