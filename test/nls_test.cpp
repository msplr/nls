#include <gtest/gtest.h>
#include <random>

#include <nls/levenberg_marquardt.h>

#include "nls/eigen_fmt.h"

//------------------------------------------------------------------------------
// Test problem from paper
//------------------------------------------------------------------------------

static double model(double t, const Eigen::VectorXd& params)
{
    const double x1 = params(0);
    const double x2 = params(1);
    const double x3 = params(2);
    const double x4 = params(3);

    return x3 * exp(x1 * t) + x4 * exp(x2 * t);
}

static Eigen::MatrixXd jacobian(double t, const Eigen::VectorXd& params)
{
    const double x1 = params(0);
    const double x2 = params(1);
    const double x3 = params(2);
    const double x4 = params(3);

    Eigen::MatrixXd res{{-x3 * t * exp(x1 * t), -x4 * t * exp(x2 * t), -exp(x1 * t), -exp(x2 * t)}};
    return res;
}

static std::pair<std::vector<double>, std::vector<double>> getData(const Eigen::VectorXd& x_star, size_t N = 200)
{
    std::random_device rd{};
    std::mt19937 gen{rd()};
    std::normal_distribution<> errorDist{0, 0.001};
    std::vector<double> times;
    std::vector<double> values;
    const double t0 = 0.0;
    const double t1 = 2.0;
    double dt = (t1 - t0) / (N - 1);
    double t = 0;
    for (double t = t0; t < t1; t += dt) {
        values.push_back(model(t, x_star) + errorDist(gen));
        times.push_back(t);
    }
    return {times, values};
}

struct SingleResidual : public nls::AutoDiffResidual<SingleResidual, /* residuals */ 1, /* params */ 4> {
    SingleResidual(double time, double value)
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

TEST(NlsTest, EvaluateResidual)
{
    const Eigen::VectorXd params{{-4, -5, 4, -4}};
    const double t = 1.0;

    auto val0 = model(t, params);
    auto jac0 = jacobian(t, params);
    spdlog::info("val0: {} jac0: {}", val0, jac0);

    SingleResidual res(t, 0.0);
    Eigen::MatrixXd jac1;
    auto residual = res.value(params, &jac1);
    spdlog::info("val1: {} jac1: {}", -residual(0), jac1);

    EXPECT_NEAR(val0, -residual(0), 1e-6);
    EXPECT_TRUE(jac0.isApprox(jac1, 1e-6));
}

TEST(NlsTest, ExampleProblemRobustLoss)
{
    const Eigen::VectorXd x_star{{-4, -5, 4, -4}};
    auto [times, values] = getData(x_star);

    nls::CostFunction prob;
    for (size_t i = 0; i < times.size(); i++) {
        prob.addResidual(new SingleResidual(times[i], values[i]), new nls::CauchyLoss());
    }

    Eigen::VectorXd x{{-1, -2, 1, -1}};
    // Eigen::VectorXd x = x_star + 0.5 * Eigen::VectorXd::Random(4);

    auto res = nls::solve(prob, x);
    EXPECT_TRUE(res == nls::Status::Converged);
    EXPECT_TRUE(x.isApprox(x_star, 1e-1));
    // EXPECT_TRUE(x.isApprox(x_star, 1e-6));

    spdlog::info("state: {}", res);
    spdlog::info("x:      {}", x.transpose());
    spdlog::info("x_star: {}", x_star.transpose());
}


struct FullResidual : public nls::AutoDiffResidualDynamic<FullResidual> {
    FullResidual(std::vector<double> time, std::vector<double> value)
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

TEST(NlsTest, ExampleProblem)
{
    const Eigen::VectorXd x_star{{-4, -5, 4, -4}};
    auto [times, values] = getData(x_star);

    nls::CostFunction prob;
    prob.addResidual(new FullResidual(times, values), nullptr);

    Eigen::VectorXd x{{-1, -2, 1, -1}};
    // Eigen::VectorXd x = x_star + 0.5 * Eigen::VectorXd::Random(4);

    auto res = nls::solve(prob, x);
    EXPECT_TRUE(res == nls::Status::Converged);
    // EXPECT_TRUE(x.isApprox(x_star, 1e-6));

    spdlog::info("state: {}", res);
    spdlog::info("x:      {}", x.transpose());
    spdlog::info("x_star: {}", x_star.transpose());
}
