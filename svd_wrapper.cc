
// we are not debugging svd so just never set this
#define NDEBUG

#include <Eigen/Dense>
#include <Eigen/SVD>

using namespace Eigen;

__attribute__((optimize("O3"))) MatrixXf getSVD_U(const MatrixXf &inp) {
  return inp.bdcSvd(Eigen::ComputeFullU).matrixU();
}
