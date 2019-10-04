#include "catch.hpp"

#include <Eigen/Dense>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>

#include "lookup.hpp"
#include "policy.hpp"

#include "pre_processing.hpp"

static boost::random::normal_distribution<float> nd;
static boost::random::mt19937 rng(0);

using namespace certified_cosine;

static auto random_matrix(int rows, int cols) {
  return Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Zero(rows, cols)
      .unaryExpr([&](auto f) -> float { return nd(rng); })
      .eval();
}

template <typename matrix_t,
          typename vector_t = Eigen::Matrix<typename matrix_t::Scalar, matrix_t::ColsAtCompileTime, 1>>
class lookup_naive {
  typedef typename matrix_t::Scalar float_t;

  typedef matrix_t MatD;
  typedef vector_t VecD;

 private:
  matrix_t matrix;

 public:
  lookup_naive(const matrix_t &mat) : matrix(mat) {}

  template <typename policy_t>
  void lookup(const VecD &x, policy_t &policy) {
    auto nkey = x;
    nkey /= nkey.norm();
    for (int i = 0; i < matrix.rows(); i++) {
      policy.expand(i, matrix.row(i).dot(nkey));
    }
  }
};

TEST_CASE("naive lookup", "[lookup]") {
  Eigen::Matrix<float, Eigen::Dynamic, 10, Eigen::RowMajor> mat = random_matrix(100, 10);

  lookup_naive<Eigen::Ref<decltype(mat)>> ln(mat);

  CountExpandPolicy<OneBestPolicy<float>> p;
  Eigen::Matrix<float, 10, 1> x = random_matrix(10, 1);

  ln.lookup(x, p);
  REQUIRE(p.count == 100);
  REQUIRE(p.id >= 0);
}

TEST_CASE("proof lookup", "[lookup]") {
  // build a small random matrix that we can run the proof system on

  auto testMat = random_matrix(1000, 15);
  auto norms = testMat.rowwise().norm().eval();
  testMat.array().colwise() /= norms.array();

  dynamic_storage<float> storage;
  preprocess<float>(testMat, storage);

  // check that the proof distances are set correctly
  for (int j = 0; j < 50; j++) {
    auto v = storage.get_vertex(j);
    float pd = v.proof_distance(&storage);
    for (auto e : v.outgoing_edges(&storage)) {
      REQUIRE(e.score >= pd);
    }
  }

  LookupCertifiedCosine<decltype(storage), Eigen::Ref<decltype(testMat)>> lp(testMat, &storage);

  CountExpandPolicy<OneBestPolicy<float>> p;
  Eigen::Matrix<float, Eigen::Dynamic, 1> x = random_matrix(15, 1);
  x /= x.norm();

  lp.lookup(x, p);

  REQUIRE(p.count > 0);
}

TEST_CASE("preprocessing", "[preprocessing]") {
  auto testMat = random_matrix(1123, 15);
  auto norms = testMat.rowwise().norm().eval();
  testMat.array().colwise() /= norms.array();

  dynamic_storage<float> storage;
  preprocess<float>(testMat, storage);

#ifdef CERTIFIEDCOSINE_WEIGHT_DIST
#error "nope"
  // for(int i = 0; i < storage.size(); i++) {
  //   auto vx = storage.get_vertex(i);
  //   float m = 1;
  //   for(int j = 0; j < vx.size(&storage); j++) {
  //     REQUIRE(vx.neighbor_dist(&storage, j).score <= m);
  //     m = vx.neighbor_dist(&storage, j).score;
  //   }
  // }
#endif
}
