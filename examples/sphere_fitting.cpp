#include <spdlog/spdlog.h>
#include <spdlog/fmt/ostr.h>
#include <Eigen/Core>

#include "nls/levenberg_marquardt.h"
#include "nls/eigen_fmt.h"

//------------------------------------------------------------------------------
// Sphere fitting problem
//------------------------------------------------------------------------------

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
1, -2, 3,
0, 0, 1,
// 0, 0, 0, // why is this outlier a problem?
};
const int numObservations = sizeof(data) / (3*sizeof(double));
// clang-format on

struct SphereResidual : public nls::AutoDiffResidual<SphereResidual, 1, 4> {
    SphereResidual(double x, double y, double z)
        : x_(x)
        , y_(y)
        , z_(z)
    {
    }
    template <typename T>
    bool operator()(const T* p, T* residual) const
    {
        residual[0] = p[0] - sqrt(pow(p[1] - x_, 2) + pow(p[2] - y_, 2) + pow(p[3] - z_, 2));
        return true;
    }

private:
    const double x_;
    const double y_;
    const double z_;
};

int main()
{
    nls::CostFunction problem;

    spdlog::info("Num Observations: {}", numObservations);
    for (int i = 0; i < numObservations; ++i) {
        const double* obs = &data[3 * i];
        auto* res = new SphereResidual(obs[0], obs[1], obs[2]);
        // problem.addResidual(res, new nls::L2Loss());
        problem.addResidual(res, new nls::CauchyLoss());
        // problem.addResidual(res, new nls::HuberLoss());
        // problem.addResidual(res, new nls::SoftL1Loss());
    }

    Eigen::VectorXd x_star{{1.0, 1.0, 2.0, -3.0}};
    Eigen::VectorXd x{{0.1, 0.0, 0.0, 0.0}};

    spdlog::info("x0:  {}", x.transpose());

    auto res = nls::solve(problem, x);

    spdlog::info("state: {}", res);
    spdlog::info("x:      {}", x.transpose());
    spdlog::info("x_star: {}", x_star.transpose());
    return 0;
}

