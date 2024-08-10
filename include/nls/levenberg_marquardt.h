#pragma once

#include <cmath>
#include <algorithm>

#include <Eigen/Dense>
#include <unsupported/Eigen/AutoDiff>

#include <spdlog/spdlog.h>
#include <spdlog/fmt/ostr.h>

namespace nls {

// Interface classes

class IResidual {
public:
    virtual Eigen::VectorXd value(const Eigen::VectorXd& params, Eigen::MatrixXd* jacobian) = 0;
    virtual size_t size() = 0;
    virtual ~IResidual() = default;
};

class ILoss {
public:
    virtual void eval(double residual, double* derivatives) = 0;
    virtual ~ILoss() = default;
};

class CostFunction {
public:
    void addResidual(IResidual* resv, ILoss* loss = nullptr);
    Eigen::VectorXd evaluate(const Eigen::VectorXd& params, Eigen::MatrixXd* jacobian = nullptr);

private:
    void applyLoss(ILoss& loss, Eigen::VectorXd& residual, Eigen::MatrixXd* jacobian = nullptr);

    std::vector<std::unique_ptr<IResidual>> residuals_;
    std::vector<std::unique_ptr<ILoss>> lossFunctions_;
};

enum class Status {
    Converged,
    IterationsLimit,
    Init
};
std::string toString(Status s);
std::ostream& operator<<(std::ostream& os, Status s);

/// Non-linear Least Squares Solver
Status solve(CostFunction& costFunc, Eigen::VectorXd& x);

// Helper classes for writing residuals using Eigen AutoDiff
template <typename DERIVED, int NUM_RESIDUALS = 1, int NUM_PARAMS = 1>
class AutoDiffResidual : public IResidual {
public:
    using ParamVector = Eigen::Matrix<double, NUM_PARAMS, 1>;
    using ADScalar = Eigen::AutoDiffScalar<ParamVector>;
    using ADResiduals = Eigen::Matrix<ADScalar, NUM_RESIDUALS, 1>;
    using ADParams = Eigen::Matrix<ADScalar, NUM_PARAMS, 1>;

    Eigen::VectorXd value(const Eigen::VectorXd& params, Eigen::MatrixXd* jacobian) override
    {
        DERIVED& functor = *static_cast<DERIVED*>(this); // CRTP in action

        Eigen::VectorXd res(NUM_RESIDUALS);

        assert(NUM_PARAMS == params.rows());
        if (jacobian == nullptr) {
            functor(params.data(), res.data());
            return res;
        } else {
            ADParams paramsAd = params;

            // seed Autodiff parameter vector
            for (int i = 0; i < NUM_PARAMS; i++) {
                paramsAd[i].derivatives() = Eigen::VectorXd::Unit(NUM_PARAMS, i);
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
    using ADScalar = Eigen::AutoDiffScalar<Eigen::VectorXd>;
    using ADVector = Eigen::Matrix<ADScalar, Eigen::Dynamic, 1>;

    Eigen::VectorXd value(const Eigen::VectorXd& params, Eigen::MatrixXd* jacobian) override
    {
        DERIVED& functor = *static_cast<DERIVED*>(this); // CRTP in action

        int numResiduals = size();
        Eigen::VectorXd res(numResiduals);

        if (jacobian == nullptr) {
            functor(params.data(), res.data());
            return res;
        } else {
            ADVector paramsAd = params;

            // seed Autodiff parameter vector
            for (int i = 0; i < params.rows(); i++) {
                paramsAd[i].derivatives() = Eigen::VectorXd::Unit(params.rows(), i);
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

// Loss functions
struct L2Loss : public nls::ILoss {
    ~L2Loss() override = default;
    void eval(double residual, double* derivatives) override
    {
        derivatives[0] = residual;
        derivatives[1] = 1.0;
    }
};

struct CauchyLoss : public nls::ILoss {
    CauchyLoss(double c = 1.0)
        : c_(c)
    {
    }
    ~CauchyLoss() override = default;
    void eval(double residual, double* derivatives) override
    {
        derivatives[0] = c_ * log(1 + residual / c_);
        derivatives[1] = 1.0 / (1 + residual / c_);
    }
    double c_;
};

struct HuberLoss : public nls::ILoss {
    ~HuberLoss() override = default;
    void eval(double residual, double* derivatives) override
    {
        if (residual > 1.0) {
            derivatives[0] = 2.0 * sqrt(residual) - 1;
            derivatives[1] = 1.0 / sqrt(residual);
        } else {
            derivatives[0] = residual;
            derivatives[1] = 1.0;
        }
    }
};

struct SoftL1Loss : public nls::ILoss {
    SoftL1Loss(double c = 1.0)
        : c_(c)
    {
    }
    ~SoftL1Loss() override = default;
    void eval(double residual, double* derivatives) override
    {
        derivatives[0] = c_ * 2.0 * (sqrt(1 + residual / c_) - 1);
        derivatives[1] = 1.0 / sqrt(1 + residual / c_);
    }
    double c_;
};

} // namespace nls

template <> struct fmt::formatter<nls::Status> : fmt::ostream_formatter {};
