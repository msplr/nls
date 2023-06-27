#include "levenberg_marquardt.h"

#include <Eigen/Dense>
#include <cmath>
#include <algorithm>

#include <spdlog/spdlog.h>
#include <spdlog/fmt/ostr.h>

namespace nls {

void CostFunction::addResidual(IResidual* res, ILoss* loss)
{
    residuals_.emplace_back(res);
    lossFunctions_.emplace_back(loss);
}

Eigen::VectorXd CostFunction::evaluate(const Eigen::VectorXd& params, Eigen::MatrixXd* jacobian)
{
    // TODO: add support for robust loss functions

    Eigen::VectorXd cost;
    Eigen::MatrixXd jac;
    Eigen::VectorXd val;

    size_t numResiduals = 0;
    for (auto& res : residuals_) {
        numResiduals += res->size();
    }
    cost.resize(numResiduals);
    if (jacobian != nullptr) {
        jacobian->resize(numResiduals, params.size());
    }

    int row = 0;
    int i = 0;
    for (auto& residual : residuals_) {
        Eigen::MatrixXd* jacp = nullptr;
        int n = residual->size();

        if (jacobian != nullptr) {
            jac.resize(n, params.size());
            jacp = &jac;
        }

        val = residual->value(params, jacp);

        if (lossFunctions_[i] != nullptr)
            applyLoss(*lossFunctions_[i], val, jacp);

        cost.segment(row, n) = val;
        if (jacobian != nullptr) {
            jacobian->middleRows(row, n) = jac;
        }

        row += n;
        i++;
    }

    return cost;
}

void CostFunction::applyLoss(ILoss& loss, Eigen::VectorXd& residual, Eigen::MatrixXd* jacobian)
{

    // TODO: it is more complicated than this...
    double derivatives[2];
    for (int i = 0; i < residual.size(); i++) {
        // residual must be positive for loss function
        if (residual(i) < 0.0) {
            residual(i) = -residual(i);
            if (jacobian != nullptr)
                jacobian->row(i) = -jacobian->row(i);
        }

        loss.eval(residual(i), derivatives);
        residual(i) = derivatives[0];
        if (jacobian != nullptr) {
            jacobian->row(i) = derivatives[1] * jacobian->row(i);
        }
    }
}

std::string toString(Status s)
{
    switch (s) {
        case Status::Converged:
            return "Converged";
        case Status::IterationsLimit:
            return "IterationsLimit";
        case Status::Init:
            return "Init";
        default:
            return "Stauts::UNKNOWN";
    }
}

std::ostream& operator<<(std::ostream& os, Status s)
{
    os << toString(s);
    return os;
}

// Levenberg-Marquardt Algorithm
// From "Methods for Non-Linear Least Squares Problems" - K. Madsen/H. B. Nielsen/O. Tingleff (2004)
Status solve(CostFunction& costFunc, Eigen::VectorXd& x)
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
    double gain_ratio;
    double cost;
    const double eps1 = 1e-8;
    const double eps2 = 1e-8;

    // initialization
    f = costFunc.evaluate(x, &J);
    g.noalias() = J.transpose() * f;

    double mu = 1e+6 * J.maxCoeff();
    // double mu = 1e-4;
    double nu = 2.0;

    for (int k = 0; k < MAX_ITER; k++) {
        // hessian H = J'J + mu I
        H.noalias() = J.transpose() * J;
        H.diagonal().array() += mu;

        // solve (J'J + mu I) dx = -g
        step = H.llt().solve(-g); // TODO: Cholesky might be numerically unstable

        if (step.norm() < eps2 * (x.norm() + eps2)) {
            spdlog::info("step criterion: step.norm() < eps2 * (x.norm() + eps2) : {:.2e} < {:.2e}", step.norm(), eps2 * (x.norm() + eps2));
            status = Status::Converged;
            break;
        }

        x_new = x + step;

        f_new = costFunc.evaluate(x_new);

        const double new_cost = f_new.dot(f_new);
        cost = f.dot(f);

        gain_ratio = (cost - new_cost) / step.dot(mu * step - g);

        if (k == 0)
            spdlog::info("iter    cost      change       rho        mu       |step|      |g|");
        spdlog::info("{:>3}: {: .2e}  {: .2e}  {: .2e}  {: .2e}  {: .2e}  {: .2e}",
                     k, cost, cost - new_cost, gain_ratio, mu, step.norm(), g.norm());

        if (gain_ratio > 0) {
            // step is acceptable
            x = x_new;
            cost = new_cost;
            f = costFunc.evaluate(x, &J);

            // gradient g = J'f
            g.noalias() = J.transpose() * f;
            if (g.lpNorm<Eigen::Infinity>() < eps1) {
                spdlog::info("gradient criterion: g.linfNorm() < eps1 : {:.2e} < {:.2e}", g.lpNorm<Eigen::Infinity>(), eps1);
                status = Status::Converged;
                break;
            }

            mu = mu * std::max(1.0 / 3.0, 1.0 - pow(2 * gain_ratio - 1, 3));
            nu = 2;
        } else {
            mu = nu * mu;
            nu = 2 * nu;
        }
        if (k == MAX_ITER - 1) {
            status = Status::IterationsLimit;
        }
    }
    return status;
}

} // namespace nls
