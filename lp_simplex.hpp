
/**
 * Written By Matthew Francis-Landau (2019)
 *
 * Implement a simplex linear programming solver
 */

#ifndef _CERTIFIEDCOSINE_LP_SIMPLEX
#define _CERTIFIEDCOSINE_LP_SIMPLEX

#include <Eigen/Dense>
#include <vector>

#ifdef DEBUG_SIMPLEX_PRINT
#include <iostream>
#endif

namespace certified_cosine {

using namespace std;

/**
 * solving the LP using the simplex method.  This is unable to handle the ||x|| <= 1 constraint
 * as we are moving along the edge of the walls instead of having the value be zero.
 * This needs to identify the cases where it has already left the unit ball and then project back into the ball
 * or then have this check if it is still inside of the cone region
 *
 * In the case that a new constraint is added, that is currently violated, then this is going to have to identify
 * some way to undo the constraint
 * - this seems to be using a "dual simplex" method where it is optimizing over which constraints are used?4
 */
template <typename float_t>
struct LPSolverSimplex {
  // the tableau as standard with simplex, pivots performed on the table
  // directly which modifies any relevant rows/columns.
  //
  // We want to be able to solve x \in (-\inf, \inf), rather than x \in [0,
  // \inf) as is more typical with simplex.  This means that we will switch
  // the signs on the columns as required

  // column 0 is the b values for a constraint
  Eigen::Matrix<float_t, -1, -1, Eigen::RowMajor> tableau;

  // the objective that we are trying to maximize
  Eigen::Matrix<float_t, -1, 1> c;

  std::vector<bool> is_negative;  // if the current variable is negated in its value

  // optimizations to track which rows and columns are being used as the basis
  std::vector<uint> row_basis;
  std::vector<uint> col_basis;  // then this can track which values was the basis and unset it

  uint nvars;
  uint nconst;

  void resize(int nconst, int nvars) {
    if (tableau.rows() < nconst + 1 || tableau.cols() < nconst + nvars + 1) {
      decltype(tableau) nC = decltype(tableau)::Zero(nconst + 1, nvars + nconst + 1);
      nC.block(0, 0, tableau.rows(), tableau.cols()) = tableau;
      tableau.swap(nC);
      is_negative.resize(nvars);
      row_basis.resize(tableau.cols());
      col_basis.resize(tableau.rows());
    }
  }

  auto &get_tableau() { return tableau.block(0, 0, nconst + 1, nvars + nconst + 1); }

  void resize(int min_size) {
    if (tableau.rows() < min_size + 1) {
      resize(max(min_size, (int)(tableau.rows() * 2)), nvars);
    }
  }

  static constexpr float_t epsilon = .002;

  // reset the solver given the new table
  // Solve Ax <= b , c^Tx <= 1 for max c^T x
  // d = A^T c  the constraints distances as precomputed
  void load_tableau(const Eigen::Matrix<float_t, -1, -1> &A,  // these should be auto so it can be a block
                    const Eigen::Matrix<float_t, -1, 1> &b, const Eigen::Matrix<float_t, -1, 1> &c
                    /*const Eigen::Matrix<float_t, -1,  1> &d*/) {
    this->c = c;
    nvars = c.rows();  //+ 1;
    nconst = b.rows() + 1;

    assert(A.rows() == nconst - 1);
    assert(A.cols() == nvars);

    // this is going have the slack variables that are added to each of constraints
    if (tableau.rows() < nconst + 1 || tableau.cols() < nvars + nconst + 1) {
      tableau.resize(nconst + 1, nvars + nconst + 1);
      row_basis.resize(tableau.cols());
      col_basis.resize(tableau.rows());
    }
    is_negative.clear();
    is_negative.resize(nvars);
    tableau.setZero();  // faster if we don't clear maybe

    // b is in the far left column, (rather than the far right like usually drawn)
    // column 1 is the d values

    assert((b.array() > 0).all());

    tableau.block(2, 0, nconst - 1, 1) = b;
    tableau.block(0, 1, 1, nvars) = -c.transpose();
    tableau.block(1, 1, 1, nvars) = c.transpose();  // set c^T x <= 1
    tableau(1, 0) = 1;
    tableau.block(2, 1, nconst - 1, nvars) = A;
    tableau.block(1, 1 + nvars, nconst, nconst).setIdentity();

    for (uint i = 0; i <= nvars; i++) {
      row_basis[i] = 0;
    }
    for (uint i = 0; i < nconst; i++) {
      col_basis[i + 1] = 1 + nvars + i;
      row_basis[1 + nvars + i] = i + 1;
    }
  }

  void switch_sign(uint idx) {
    tableau.col(idx + 1) *= -1;
    is_negative[idx].flip();  // ^= true;  // ..... the joys of using vector<bool>
  }

  void add_constraint(const Eigen::Matrix<float_t, -1, 1> &a, float_t b, float_t d = 0.0) {
    // add the constraint Ax <= b
    assert(b > 0);
    assert(a.size() == nvars);

    nconst++;
    resize(nconst, nvars);
    tableau.col(nvars + nconst).setZero();
    tableau.block(nconst, 1 + nvars, 1, nconst).setZero();
    tableau(nconst, nvars + nconst) = 1;

    tableau(nconst, 0) = b;
    for (uint i = 0; i < nvars; i++) {
      if (is_negative[i]) {
        tableau(nconst, i + 1) = -a(i);
      } else {
        tableau(nconst, i + 1) = a(i);
      }
    }

    col_basis[nconst] = nvars + nconst;
    row_basis[nvars + nconst] = nconst;

    // transform the new constraint into the currently active basis

    auto self = tableau.block(nconst, 0, 1, nvars + nconst);

    for (uint i = 0; i < nvars; i++) {
      // check if the current column is a basis, then we are going to subtract it from our row
      const uint rb = row_basis[i + 1];
      float_t v = self(0, i + 1);
      if (rb && col_basis[rb] == i + 1 && v != 0) {
        self -= tableau.block(rb, 0, 1, self.cols()) * v;
      }
    }
  }

  inline float_t objective() { return tableau(0, 0); }

  bool is_constraint_tight(uint c = 0) { return col_basis[row_basis[c]] == c; }

  Eigen::Matrix<float_t, 1, -1> get_x() {
    Eigen::Matrix<float_t, 1, -1> ret(nvars);
    get_x(ret);
    return ret;
  }

  void get_x(Eigen::Matrix<float_t, 1, -1> &ret) {
    for (uint i = 0; i < nvars; i++) {
      const uint rb = row_basis[i + 1];
      if (rb && col_basis[rb] == i + 1) {
        assert(abs(tableau(0, i + 1)) < .0005);
        assert(tableau(rb, i + 1) == 1);
        ret(i) = tableau(rb, 0) * (is_negative[i] ? -1 : 1);
      } else {
        ret(i) = 0;
      }
    }
  }

  Eigen::Matrix<float_t, 1, -1> get_x_vec() {
    Eigen::Matrix<float_t, 1, -1> v1(nvars);
    get_x(v1);
    return v1;
  }

  bool check_solution() {
    // check that the solution to the simplex respects all of the constraints
    return (tableau.block(1, 0, nconst, 1).array() >= -.005).all();
  }

  void do_pivot(const int pivotRow, const int pivotColumn) {
#ifdef DEBUG_SIMPLEX_PRINT
    cout << "pivot: " << pivotRow << " " << pivotColumn << " " << objective() << endl;
#endif

    // divide out the row
    float_t div = tableau(pivotRow, pivotColumn);
    const int ncols = nvars + nconst + 1;
    auto prow = tableau.block(pivotRow, 0, 1, ncols);
    assert(div != 0);
    prow /= div;

    // track where the basis is
    row_basis[col_basis[pivotRow]] = 0;
    col_basis[pivotRow] = pivotColumn;
    row_basis[pivotColumn] = pivotRow;

    for (uint i = 0; i <= nconst; i++) {  // there might be extra rows that are stored
      if (i != pivotRow) {
        float_t val = tableau(i, pivotColumn);
        if (val != 0) {
          tableau.block(i, 0, 1, ncols) -= prow * val;
        }
      }
    }

    assert(tableau(0, pivotColumn) == 0);
    assert(tableau(pivotRow, pivotColumn) == 1);
  }

  bool run_dual_simplex() {
    // this descreases the value to ensure that all of the constriants are satasified
    int pivotColumn, pivotRow;

    while (true) {
      float_t val = numeric_limits<float_t>::infinity();
      pivotRow = -1;
      for (uint i = 0; i < nconst; i++) {
        float_t v = tableau(i + 1, 0);
        if (v < val) {
          val = v;
          pivotRow = i;
        }
      }
      // then something has gone wrong.  There seems to be some floating point error that accumulates, and causes
      // problems
      if (pivotRow == -1) return false;
      pivotRow++;

      if (val >= 0) break;  // done if we have nothing worse that epsilon

      val = numeric_limits<float_t>::infinity();
      pivotColumn = -1;
      for (uint i = 0; i < nvars + nconst; i++) {
        float_t n = tableau(0, i + 1);
        float_t d = tableau(pivotRow, i + 1);
        if (d >= 0) continue;
        float_t v = -n / d;
        if (v < val) {
          val = v;
          pivotColumn = i;
        }
      }
      if (pivotColumn == -1) return false;
      pivotColumn++;

      do_pivot(pivotRow, pivotColumn);
    }

#ifdef CERTIFIEDCOSINE_DEBUG
    assert(check_solution());
#endif
    return true;
  }

  void run_simplex() {
    int pivotColumn, pivotRow;

    while (objective() < 1 - epsilon) {
      assert((tableau.block(1, 0, nconst, 1).array() >= -.005).all());

      float_t val = numeric_limits<float_t>::infinity();
      pivotColumn = -1;
      for (uint i = 0; i < nvars; i++) {
        float_t v = tableau(0, i + 1);
        if (v > 0) v *= -1;
        if (v < val) {
          val = v;
          pivotColumn = i;
        }
      }
      for (uint i = 0; i < nconst; i++) {
        float_t v = tableau(0, nvars + 1 + i);
        if (v < val) {
          val = v;
          pivotColumn = i + nvars;
        };
      }
      assert(pivotColumn != -1);

      if (val > -epsilon) return;  // the amount we can improve is not that much

      // switch the sign of the cases where this is a "negative" value
      if (pivotColumn < nvars && tableau(0, pivotColumn + 1) > 0) {
        switch_sign(pivotColumn);
      }

      pivotColumn++;

      float_t val1 = val;

      // find the pivotRow
      val = numeric_limits<float_t>::infinity();
      pivotRow = -1;
      for (uint i = 1; i <= nconst; i++) {
        float_t d = tableau(i, pivotColumn);
        float_t v = tableau(i, 0) / d;
        if (d <= 0) continue;
        assert(v > 0);
        if (v < val) {
          val = v;
          pivotRow = i;
        }
      }

      assert(pivotRow != -1);

      do_pivot(pivotRow, pivotColumn);

#ifdef CERTIFIEDCOSINE_DEBUG
      assert(check_solution());
#endif
    }
  }
};

}  // namespace certified_cosine

#endif
