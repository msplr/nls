#include <vector>
#include <random>
#include <cmath>
#include <algorithm>

#include <Eigen/Dense>
#include <unsupported/Eigen/AutoDiff>

#include <spdlog/spdlog.h>
#include <spdlog/fmt/ostr.h>

#include "nls/eigen_fmt.h"

template <typename DERIVED, int numResiudals = 1>
class AutoDiffResidual {
public:
    using Vector = Eigen::Matrix<double, Eigen::Dynamic, 1>;
    using Matrix = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>;

    using ADScalar = Eigen::AutoDiffScalar<Vector>;
    using ADVector = Eigen::Matrix<ADScalar, Eigen::Dynamic, 1>;

    Vector value(const Vector& x, Matrix* jacobian) const
    {
        if (jacobian == nullptr) {
            return static_cast<const DERIVED*>(this)->residual(x);
        } else {
            ADVector ad_x = x;

            // seed Autodiff parameter vector
            for (int i = 0; i < x.rows(); i++) {
                ad_x[i].derivatives() = Vector::Unit(x.rows(), i);
            }

            ADVector ad_res = static_cast<const DERIVED*>(this)->residual(ad_x);
            Vector res(ad_res.size());

            // fill constraint Jacobian
            jacobian->resize(ad_res.size(), x.size());
            for (int i = 0; i < ad_res.size(); i++) {
                res[i] = ad_res[i].value();
                jacobian->row(i) = ad_res[i].derivatives().transpose();
            }
            return res;
        }
    }
};

struct Residual : public AutoDiffResidual<Residual> {
    Residual(double time, double value)
        : t(time)
        , y(value)
    {
    }

    template <typename Scalar>
    Eigen::Matrix<Scalar, Eigen::Dynamic, 1> residual(const Eigen::Matrix<Scalar, Eigen::Dynamic, 1>& params) const
    {
        const Scalar& x1 = params(0);
        const Scalar& x2 = params(1);
        const Scalar& x3 = params(2);
        const Scalar& x4 = params(3);

        Eigen::Matrix<Scalar, Eigen::Dynamic, 1> res(1);
        res(0) = y - (x3 * exp(x1 * t) + x4 * exp(x2 * t));
        return res;
    }

    int size() const { return 1; }

    double t;
    double y;
};

struct Problem {
    Problem(const std::vector<double>& times, const std::vector<double>& values);

    Eigen::VectorXd residual(const Eigen::VectorXd& params, Eigen::MatrixXd* jacobian = nullptr) const;
    double model(double time, const Eigen::VectorXd& params, Eigen::VectorXd* gradient) const;

    std::vector<Residual> residuals;
};

Problem::Problem(const std::vector<double>& times, const std::vector<double>& values)
{
    if (times.size() != values.size()) {
        return;
    }

    for (int i = 0; i < times.size(); i++) {
        residuals.emplace_back(times[i], values[i]);
    }
}

Eigen::VectorXd Problem::residual(const Eigen::VectorXd& params, Eigen::MatrixXd* jacobian) const
{
    Eigen::VectorXd res(residuals.size());

    Eigen::MatrixXd jac;
    Eigen::MatrixXd* jac_p = nullptr;

    if (jacobian != nullptr) {
        jac_p = &jac;
        jac.resize(1, params.size());
        jacobian->resize(res.size(), params.size());
    }

    int i = 0;
    for (auto& residual : residuals) {
        int n = residual.size();
        res.segment(i, n) = residual.value(params, jac_p);
        if (jacobian != nullptr) {
            jacobian->middleRows(i, n) = jac;
        }
        i += n;
    }

    return res;
}

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

bool cholesky(const Eigen::MatrixXd& A, Eigen::MatrixXd& L);
Eigen::VectorXd solve(const Eigen::MatrixXd& L, const Eigen::VectorXd& b);

void test_cholesky()
{
    Eigen::MatrixXd A{{4, 12, -16},
                      {12, 37, -43},
                      {-16, -43, 98}};
    Eigen::MatrixXd L_expect{{2, 0, 0},
                             {6, 1, 0},
                             {-8, 5, 3}};

    Eigen::VectorXd b;
    b.setRandom(3);

    Eigen::MatrixXd L;
    bool posdef = cholesky(A, L);
    spdlog::info("cholesky ok: {}, posdef {}", L.isApprox(L_expect), posdef);

    Eigen::VectorXd sol = A.llt().solve(b);
    Eigen::VectorXd res = solve(L, b);

    spdlog::info("res {}  ok {}", res.transpose(), res.isApprox(sol));
    spdlog::info("sol {}", sol.transpose());
}

int main(void)
{
    std::random_device rd{};
    std::mt19937 gen{rd()};

    std::normal_distribution<> errorDist{0, 0.001};

    const Eigen::VectorXd x_star{{-4, -5, 4, -4}};

    std::vector<double> times;
    std::vector<double> values;
    for (double t = 0; t < 2; t += 0.01) {
        // double y = model(t, x_star);
        double y = model(t, x_star) + errorDist(gen);
        values.push_back(y);
        times.push_back(t);
    }

    Problem prob(times, values);

    // Levenberg-Marquardt Algorithm
    // taken from "Methods for Non-Linear Least Squares Problems" - K. Madsen/H. B. Nielsen/O. Tingleff (2004)
    Eigen::VectorXd x{{-1, -2, 1, -1}};
    // Eigen::VectorXd x = x_star + 0.5 * Eigen::VectorXd::Random(4);
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
    f = prob.residual(x, &J);
    g.noalias() = J.transpose() * f;
    mu = 1e+6 * J.maxCoeff();

    for (int k = 0; k < 100; k++) {
        // hessian H = J'J + mu I
        H.noalias() = J.transpose() * J;
        H.diagonal().array() += mu;

        // solve (J'J + mu I) dx = -g
        // step = H.llt().solve(-g);

        Eigen::MatrixXd L;
        bool posdef = cholesky(H, L);
        if (!posdef) {
            break;
        }
        step = solve(L, -g);

        if (step.norm() < eps2 * (x.norm() + eps2)) {
            spdlog::info("step criterion: step.norm() < eps2 * (x.norm() + eps2) : {} < {}", step.norm(), eps2 * (x.norm() + eps2));
            break;
        }

        x_new = x + step;

        f_new = prob.residual(x_new);

        gain_ratio = (f.dot(f) - f_new.dot(f_new)) / step.dot(mu * step - g);

        spdlog::info("k {}: Fx {}  gain_ratio {}  mu {}  step_norm {} gradient {}", k, f.dot(f), gain_ratio, mu, step.norm(), g.transpose());

        if (gain_ratio > 0) {
            // step is acceptable
            x = x_new;
            f = prob.residual(x, &J);
            // gradient g = J'f
            g.noalias() = J.transpose() * f;
            if (g.lpNorm<Eigen::Infinity>() < eps1) {
                spdlog::info("gradient criterion: g.linfNorm() < eps1 : {} < {}", g.lpNorm<Eigen::Infinity>(), eps1);
                break;
            }

            mu = mu * fmax(1.0 / 3.0, 1.0 - pow(2 * gain_ratio - 1, 3));
            nu = 2;
        } else {
            mu = nu * mu;
            nu = 2 * nu;
        }
    }
    spdlog::info("x: {}", x.transpose());
    spdlog::info("x_star: {}", x_star.transpose());

    return 0;
}
