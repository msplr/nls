#include <vector>
#include <random>
#include <cmath>
#include <algorithm>

#include <Eigen/Dense>
#include <unsupported/Eigen/AutoDiff>

#include <spdlog/spdlog.h>
#include <spdlog/fmt/ostr.h>

namespace nls {

using Vector = Eigen::Matrix<double, Eigen::Dynamic, 1>;
using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

class IResidual {
public:
    virtual Vector value(const Vector& params, Matrix* jacobian) = 0;
    virtual size_t size() = 0;
    virtual ~IResidual() = default;
};

template <typename DERIVED, int NUM_RESIDUALS = 1, int NUM_PARAMS = 1>
class AutoDiffResidual : public IResidual {
public:
    using ParamVector = Eigen::Matrix<double, NUM_PARAMS, 1>;
    using ADScalar = Eigen::AutoDiffScalar<ParamVector>;
    using ADResiduals = Eigen::Matrix<ADScalar, NUM_RESIDUALS, 1>;
    using ADParams = Eigen::Matrix<ADScalar, NUM_PARAMS, 1>;

    Vector value(const Vector& params, Matrix* jacobian) override
    {
        DERIVED& functor = *static_cast<DERIVED*>(this); // CRTP in action

        Vector res(NUM_RESIDUALS);

        assert(NUM_PARAMS == params.rows());
        if (jacobian == nullptr) {
            functor(params.data(), res.data());
            return res;
        } else {
            ADParams paramsAd = params;

            // seed Autodiff parameter vector
            for (int i = 0; i < NUM_PARAMS; i++) {
                paramsAd[i].derivatives() = Vector::Unit(NUM_PARAMS, i);
            }

            ADResiduals ad_res;
            functor(paramsAd.data(), ad_res.data());

            // fill constraint Jacobian
            jacobian->resize(NUM_RESIDUALS, NUM_PARAMS);
            for (int i = 0; i < NUM_RESIDUALS; i++) {
                res[i] = ad_res[i].value();
                jacobian->row(i).noalias() = ad_res[i].derivatives().transpose();
            }
            return res;
        }
    }

    size_t size() override { return NUM_RESIDUALS; }
};

template <typename DERIVED>
class AutoDiffResidualDynamic : public IResidual {
public:
    using ADScalar = Eigen::AutoDiffScalar<Vector>;
    using ADVector = Eigen::Matrix<ADScalar, Eigen::Dynamic, 1>;

    Vector value(const Vector& params, Matrix* jacobian) override
    {
        DERIVED& functor = *static_cast<DERIVED*>(this); // CRTP in action

        int numResiduals = size();
        Vector res(numResiduals);

        if (jacobian == nullptr) {
            functor(params.data(), res.data());
            return res;
        } else {
            ADVector paramsAd = params;

            // seed Autodiff parameter vector
            for (int i = 0; i < params.rows(); i++) {
                paramsAd[i].derivatives() = Vector::Unit(params.rows(), i);
            }

            ADVector ad_res(numResiduals);
            functor(paramsAd.data(), ad_res.data());

            // fill constraint Jacobian
            jacobian->resize(numResiduals, params.rows());
            for (int i = 0; i < numResiduals; i++) {
                res[i] = ad_res[i].value();
                jacobian->row(i).noalias() = ad_res[i].derivatives().transpose();
            }
            return res;
        }
    }
};

class ILoss {
public:
    virtual void eval(double cost, double* derivatives) = 0;
    virtual ~ILoss() = default;
};

class CostFunction {
public:
    void addResidual(IResidual* resv, ILoss* loss);
    Eigen::VectorXd residual(const Eigen::VectorXd& params, Eigen::MatrixXd* jacobian = nullptr);

private:
    std::vector<std::unique_ptr<IResidual>> residuals_;
    std::vector<std::unique_ptr<ILoss>> lossFunctions_;
};

void CostFunction::addResidual(IResidual* res, ILoss* loss)
{
    residuals_.emplace_back(res);
    lossFunctions_.emplace_back(loss);
}

Eigen::VectorXd CostFunction::residual(const Eigen::VectorXd& params, Eigen::MatrixXd* jacobian)
{
    Eigen::VectorXd cost;

    size_t numResiduals = 0;
    for (auto& res : residuals_) {
        numResiduals += res->size();
    }
    cost.resize(numResiduals);
    if (jacobian != nullptr) {
        jacobian->resize(numResiduals, params.size());
    }

    int row = 0;
    for (auto& res : residuals_) {
        int n = res->size();
        if (jacobian != nullptr) {
            Eigen::MatrixXd jac;
            jac.resize(n, params.size());
            cost.segment(row, n) = res->value(params, &jac);
            jacobian->middleRows(row, n) = jac;
        } else {
            cost.segment(row, n) = res->value(params, nullptr);
        }
        row += n;
    }

    return cost;
}

enum class Status {
    Converged,
    MaxIterationsExceeded,
    Init
};

// Levenberg-Marquardt Algorithm
// taken from "Methods for Non-Linear Least Squares Problems" - K. Madsen/H. B. Nielsen/O. Tingleff (2004)
Status solve(CostFunction& cost, Eigen::VectorXd& x)
{
    constexpr int MAX_ITER = 100;
    Status status = Status::Init;
    Eigen::VectorXd x_new;
    Eigen::VectorXd f;
    Eigen::VectorXd f_new;
    Eigen::MatrixXd J;
    Eigen::VectorXd g;
    Eigen::MatrixXd H;
    Eigen::VectorXd step;
    double mu = 0.0;
    double nu = 2.0;
    double gain_ratio;
    const double eps1 = 1e-8;
    const double eps2 = 1e-8;

    // initialization
    f = cost.residual(x, &J);
    g.noalias() = J.transpose() * f;
    mu = 1e+6 * J.maxCoeff();

    for (int k = 0; k < MAX_ITER; k++) {
        // hessian H = J'J + mu I
        H.noalias() = J.transpose() * J;
        H.diagonal().array() += mu;

        // solve (J'J + mu I) dx = -g
        step = H.llt().solve(-g);

        if (step.norm() < eps2 * (x.norm() + eps2)) {
            spdlog::info("step criterion: step.norm() < eps2 * (x.norm() + eps2) : {} < {}", step.norm(), eps2 * (x.norm() + eps2));
            status = Status::Converged;
            break;
        }

        x_new = x + step;

        f_new = cost.residual(x_new);

        gain_ratio = (f.dot(f) - f_new.dot(f_new)) / step.dot(mu * step - g);

        spdlog::info("k {}: Fx {}  gain_ratio {}  mu {}  step_norm {} gradient {}", k, f.dot(f), gain_ratio, mu, step.norm(), g.norm());

        if (gain_ratio > 0) {
            // step is acceptable
            x = x_new;
            f = cost.residual(x, &J);
            // gradient g = J'f
            g.noalias() = J.transpose() * f;
            if (g.lpNorm<Eigen::Infinity>() < eps1) {
                spdlog::info("gradient criterion: g.linfNorm() < eps1 : {} < {}", g.lpNorm<Eigen::Infinity>(), eps1);
                status = Status::Converged;
                break;
            }

            mu = mu * fmax(1.0 / 3.0, 1.0 - pow(2 * gain_ratio - 1, 3));
            nu = 2;
        } else {
            mu = nu * mu;
            nu = 2 * nu;
        }
        if (k == MAX_ITER - 1) {
            status = Status::MaxIterationsExceeded;
        }
    }
    return status;
}

} // namespace nls

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
    const Vector params{{-4, -5, 4, -4}};
    
    const double t = 1.0;
    
    auto val0 = model(t, params);
    auto jac0 = jacobian(t, params);
    spdlog::info("val0: {} jac0: {}", val0, jac0);

    TestResidual res1(t, 0.0);
    Matrix jac1;
    auto val1 = res1.value(params, &jac1);
    spdlog::info("val1: {} jac1: {}", -val1(0), jac1);
}

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
// 10, 20, 30,
};
const int numObservations = sizeof(data) / (3*sizeof(double));
// clang-format on

struct SphereResidual : public nls::AutoDiffResidual<TestResidual, 1, 4> {
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
        // residual[0] = pow(p[0], 2) - pow(p[1] - x_, 2) + pow(p[2] - y_, 2) + pow(p[3] - z_, 2);
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
    Eigen::Vector3d mean{0, 0, 0};
    for (int i = 0; i < numObservations; ++i) {
        const double* obs = &data[3 * i];
        mean += Eigen::Vector3d(obs[0], obs[1], obs[2]);
        auto* res = new SphereResidual(obs[0], obs[1], obs[2]);
        problem.addResidual(res, nullptr);
    }
    mean = mean / numObservations;

    Eigen::VectorXd x_star{{1.0, 1.0, 2.0, -3.0}};

    // none of these init params work
    // Eigen::VectorXd x{{1.0, mean[0], mean[1], mean[2]}};
    // Eigen::VectorXd x{{0.1, 0.0, 0.0, 0.0}};

    // Not even starting with the solution. There must be a bug somewhere
    Eigen::VectorXd x = x_star;

    spdlog::info("x0:  {}", x.transpose());

    auto res = nls::solve(problem, x);

    spdlog::info("state: {}", (int)res);
    spdlog::info("x:      {}", x.transpose());

    spdlog::info("x_star: {}", x_star.transpose());
}

int main()
{
    testResidual();
    testSphere();
    return 0;

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
    spdlog::info("state: {}", (int)res);
    spdlog::info("x:      {}", x.transpose());
    spdlog::info("x_star: {}", x_star.transpose());

    return 0;
}
