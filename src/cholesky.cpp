#include <cmath>
#include <Eigen/Dense>

bool cholesky(const Eigen::MatrixXd& A, Eigen::MatrixXd& L)
{
    const int n = A.rows();
    L.setZero(n, n);
    // iter over col j
    for (int j = 0; j < n; j++) {
        double d = A(j, j) - L.row(j).head(j).squaredNorm();

        if (d <= 0) {
            return false;
        }

        L(j, j) = sqrt(d);
        // iter over row i, always: i > j
        for (int i = j + 1; i < n; i++) {
            double sum = L.row(j).head(j).dot(L.row(i).head(j));
            L(i, j) = (A(i, j) - sum) / L(j, j);
        }
    }
    return true;
}

Eigen::VectorXd solve(const Eigen::MatrixXd& L, const Eigen::VectorXd& b)
{
    const int n = b.size();

    // forward substitution
    Eigen::VectorXd y(n);
    for (int i = 0; i < n; i++) {
        y(i) = (b(i) - L.row(i).head(i).dot(y.head(i))) / L(i, i);
    }

    // backward substitution
    Eigen::VectorXd x(n);
    for (int i = n - 1; i >= 0; i--) {
        x(i) = (y(i) - L.col(i).tail(n - i - 1).dot(x.tail(n - i - 1))) / L(i, i);
    }

    return x;
}
