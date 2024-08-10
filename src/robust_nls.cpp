#include <vector>
#include <random>

#include "levenberg_marquardt.h"

#include "eigen_fmt.h"

//------------------------------------------------------------------------------
// Test problem from paper
//------------------------------------------------------------------------------

double model(double t, const Eigen::VectorXd& params)
{
    const double x1 = params(0);
    const double x2 = params(1);
    const double x3 = params(2);
    const double x4 = params(3);

    return x3 * exp(x1 * t) + x4 * exp(x2 * t);
}

Eigen::MatrixXd jacobian(double t, const Eigen::VectorXd& params)
{
    const double x1 = params(0);
    const double x2 = params(1);
    const double x3 = params(2);
    const double x4 = params(3);

    Eigen::MatrixXd res{{-x3 * t * exp(x1 * t), -x4 * t * exp(x2 * t), -exp(x1 * t), -exp(x2 * t)}};
    return res;
}

struct TestResidual : public nls::AutoDiffResidual<TestResidual, /* residuals */ 1, /* params */ 4> {
    TestResidual(double time, double value)
        : t(time)
        , y(value)
    {
    }

    template <typename Scalar>
    bool operator()(const Scalar* params, Scalar* residual)
    {
        const Scalar& x1 = params[0];
        const Scalar& x2 = params[1];
        const Scalar& x3 = params[2];
        const Scalar& x4 = params[3];
        residual[0] = y - (x3 * exp(x1 * t) + x4 * exp(x2 * t));
        return true;
    }

    double t;
    double y;
};

struct TestResidual2 : public nls::AutoDiffResidualDynamic<TestResidual2> {
    TestResidual2(std::vector<double> time, std::vector<double> value)
        : t(std::move(time))
        , y(std::move(value))
    {
        assert(t.size() == y.size());
    }

    template <typename Scalar>
    bool operator()(const Scalar* params, Scalar* residual)
    {
        const Scalar& x1 = params[0];
        const Scalar& x2 = params[1];
        const Scalar& x3 = params[2];
        const Scalar& x4 = params[3];
        for (size_t i = 0; i < t.size(); i++) {
            residual[i] = y[i] - (x3 * exp(x1 * t[i]) + x4 * exp(x2 * t[i]));
        }
        return true;
    }

    size_t size() override { return t.size(); }

private:
    std::vector<double> t;
    std::vector<double> y;
};

void testResidual()
{
    using namespace nls;
    const Eigen::VectorXd params{{-4, -5, 4, -4}};

    const double t = 1.0;

    auto val0 = model(t, params);
    auto jac0 = jacobian(t, params);
    spdlog::info("val0: {} jac0: {}", val0, jac0);

    TestResidual res1(t, 0.0);
    Eigen::MatrixXd jac1;
    auto val1 = res1.value(params, &jac1);
    spdlog::info("val1: {} jac1: {}", -val1(0), jac1);
}

void testExampleProblem()
{
    std::random_device rd{};
    std::mt19937 gen{rd()};

    std::normal_distribution<> errorDist{0, 0.001};

    const Eigen::VectorXd x_star{{-4, -5, 4, -4}};

    nls::CostFunction prob;

    std::vector<double> times;
    std::vector<double> values;
    for (double t = 0; t < 2; t += 0.01) {
        double y = model(t, x_star) + errorDist(gen);
        values.push_back(y);
        times.push_back(t);
        // prob.addResidual(new TestResidual(t, y), nullptr);
    }

    prob.addResidual(new TestResidual2(times, values), nullptr);

    Eigen::VectorXd x{{-1, -2, 1, -1}};
    // Eigen::VectorXd x = x_star + 0.5 * Eigen::VectorXd::Random(4);

    auto res = nls::solve(prob, x);
    spdlog::info("state: {}", res);
    spdlog::info("x:      {}", x.transpose());
    spdlog::info("x_star: {}", x_star.transpose());
}

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

void testSphere()
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
}

int main()
{
    testResidual();
    testSphere();
    // testExampleProblem();

    return 0;
}
