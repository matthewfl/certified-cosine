#include "catch.hpp"

#include "lp_project.hpp"
#include "lp_simplex.hpp"

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>

static boost::random::normal_distribution<float> nd;
static boost::random::mt19937 rng(0);

static auto random_matrix(int rows, int cols) {
  return Eigen::MatrixXf::Zero(rows, cols).unaryExpr([&](auto f) -> float { return nd(rng); });
}

TEST_CASE("fast half space intersection") {
  using namespace certified_cosine;

  LPSolverProject<float> solver;

  solver.init(3);  // init the dimention
  solver.c = decltype(solver.c)::Zero(3);
  solver.c << 1, 2, 3;
  solver.c /= solver.c.norm();
  solver.reset_x();

  REQUIRE(solver.check_vector(solver.x));

  // this is a really bad interface
  solver.resize(1);
  solver.A.row(0) << 1, 2, 4;
  solver.A.row(0) /= solver.A.row(0).norm();
  solver.b(0) = .7;
  solver.num_rows++;

  REQUIRE(!solver.check_vector(solver.x));

  solver.optimize();

  REQUIRE(solver.check_vector(solver.x));
}

TEST_CASE("simplex") {
  using namespace certified_cosine;
  LPSolverSimplex<float> solver;

  SECTION("basic") {
    Eigen::Matrix<float, -1, 1> c(3);
    c << 1, 2, 3;
    c /= c.norm();

    Eigen::Matrix<float, -1, -1> A(1, 3);
    A << 1, 2, 4;
    A.row(0) /= A.row(0).norm();

    Eigen::Matrix<float, -1, 1> b(1);
    b << .7;

    Eigen::Matrix<float, -1, 1> d = A * c;

    solver.load_tableau(A, b, c);

    REQUIRE(solver.objective() == 0.0f);
    REQUIRE(solver.get_x().isZero());

    solver.resize(4);
  }

  SECTION("solve few variables") {
    Eigen::Matrix<float, -1, -1> A = random_matrix(5, 10);
    A.array().colwise() /= A.rowwise().norm().array();  // norm the arrays
    Eigen::Matrix<float, -1, 1> c = random_matrix(10, 1).normalized();
    Eigen::Matrix<float, -1, 1> b = random_matrix(5, 1).array().abs();
    Eigen::Matrix<float, -1, 1> d = A * c;

    solver.load_tableau(A, b, c);

    REQUIRE(solver.objective() == 0.0f);
    REQUIRE(solver.get_x().isZero());
    REQUIRE(solver.get_x().size() == 10);

    solver.run_simplex();

    float obj1 = solver.objective();

    REQUIRE(obj1 > .9);  // should be about 1
    REQUIRE(obj1 <= Approx(1));
    REQUIRE(solver.check_solution());
    REQUIRE(!solver.get_x().isZero());

    SECTION("add constraint") {
      Eigen::Matrix<float, -1, 1> a = solver.get_x_vec();
      a /= a.norm();
      solver.add_constraint(a, .7, a.dot(c));
      REQUIRE(!solver.check_solution());

      // run the dual simplex method

      solver.run_dual_simplex();

      float obj2 = solver.objective();
      REQUIRE(obj2 <= Approx(obj1));
      REQUIRE(obj2 > 0);

      solver.run_simplex();

      float obj3 = solver.objective();
      REQUIRE(obj3 <= Approx(obj1));  // these may all just be equal
      REQUIRE(obj3 >= Approx(obj2));

      REQUIRE(!solver.get_x().isZero());
    }
  }
}
