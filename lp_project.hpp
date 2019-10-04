
/**
 * Written By Matthew Francis-Landau (2019)
 *
 * Implement convex projection for locating a point
 */

#ifndef _CERTIFIEDCOSINE_LP_PROJECT
#define _CERTIFIEDCOSINE_LP_PROJECT

#include <Eigen/Dense>
#include <vector>

#include <boost/numeric/interval.hpp>
#include "feroundingmode.h"

#include "constants.hpp"

#ifdef CERTIFIEDCOSINE_DEBUG_LP_PRINT
#include <iostream>
#endif

namespace certified_cosine {

using namespace std;

/**
 * attempt to solve the linear program by projecting into the convex instead
 * of making it solve the lienar program.  This lets us have |x| <= 1 as a
 * constraint.
 *
 * We can also /normalize/ the vector X after every operation, and attempt to
 * find an X that is on the surface of the unit sphere (which is the point in x
 * \in S_t that we are actually interested in finding.
 */
template <typename float_t, int matrix_cols = Eigen::Dynamic>
struct LPSolverProject {
  // solve for Ax <= b with a point close to c
  Eigen::Matrix<float_t, -1, matrix_cols, Eigen::RowMajor>
      A;                            // summary vectors that we have fully expanded neighborhoods
  Eigen::Matrix<float_t, -1, 1> b;  // the b vectors which represents the size of the neighborhoods
  Eigen::Matrix<float_t, matrix_cols, 1>
      c;  // the starting location for projection (also the target vector that we are trying to find a point near)
  Eigen::Matrix<float_t, matrix_cols, 1> x;  // the current X value that we are evaluating

  Eigen::Matrix<float_t, matrix_cols, 1> old_x;  // old value of x for checking if the sequence is converging

  Eigen::Matrix<float_t, -1, 1> d;  // cache of the Ac values, not used by the project algorithm

  // the number of rows of A that are currently used
  uint num_rows = 0;

  void init(int ndim) {
    A = decltype(A)::Zero(0, ndim);
    b = decltype(b)::Zero(0);
    d = decltype(d)::Zero(0);
    resize(10);
    num_rows = 0;
  }

  void resize(int min_size) {
    if (A.rows() < min_size) {
      int s = A.rows() * 2;
      if (s < min_size) s = min_size;
      decltype(A) nA = decltype(A)::Zero(s, A.cols());
      decltype(b) nb = decltype(b)::Zero(s);
      decltype(d) nd = decltype(d)::Zero(s);
      nA.topRows(A.rows()) = A;
      nb.topRows(b.rows()) = b;
      nd.topRows(d.rows()) = d;
      A.swap(nA);
      b.swap(nb);
      d.swap(nd);
    }
  }

  float_t proof_distance_cosim = 0;  // the distance in cosine of S_t
  float_t located_cosim = -1;        // the distance of x \in S_t that we have currently located

  // the extra gap is to try and push the system to find something that is is
  // further away the idea being that if finds something that is right along
  // the boundary is not going to end up finding something that provides that
  // much additional information
  static constexpr float_t extra_gap_subtract = 0;
  static constexpr float_t extra_gap_mul = .9994;

#ifdef CERTIFIEDCOSINE_DEBUG
  uint optimize_calls = 0;
  uint optimize_loops = 0;
  uint num_check_innerproducts = 0;
#endif

  typedef Eigen::Matrix<float_t, -1, 1> evec;

  bool check_vector(const evec &v) {
    if (num_rows == 0) return true;
    auto Ap = A.topRows(num_rows);
    auto bp = b.topRows(num_rows);
    auto rr = (bp - (Ap * v)).array().eval();
    assert(!rr.hasNaN());
    auto r1 = (rr >= 0).eval();
    auto r2 = (bp.array() >= (Ap * v).array().eval()).eval();
    assert((r1 == r2).all());
    return r1.all();
  }

  void reset_x() {
    // unclear that this should reset the vector.  if this is just projecting in the set, then when it adds something
    // new that it needs to avoid, it should be able to just pick up where it left off??
    x = c;
  }

  void reset() {
    num_rows = 0;
#ifdef CERTIFIEDCOSINE_DEBUG
    num_check_innerproducts = optimize_calls = optimize_loops = 0;
#endif
  }

  bool optimize(bool on_sphere_surface = true) {
    using namespace std;

#ifdef CERTIFIEDCOSINE_DEBUG
    optimize_calls++;
#endif

    // in the case that we are trying to stay on the surface of the sphere,
    // then we are conceptually expanding the height of the initial vector.
    // We track the what the lenght of that initial vector would have been as
    // it helps ensure that we don't just run off forever
    float_t alpha_height = 1;

    bool violated_constraint;
    uint idx;
    uint64_t recheck_constraints;
    uint loop_times = 0;

    located_cosim = 1.2;

    // for checking if the x value is converging, otherwise we have a problem
    float_t old_x_distances = 1;
    if (!on_sphere_surface) old_x = x;

    using namespace boost::numeric;
    using namespace interval_lib;
    using namespace compare::certain;

    typedef interval<float_t> FF;

    auto opt_constraint = [&](const int i) -> bool {
#ifdef CERTIFIEDCOSINE_DEBUG
      num_check_innerproducts++;
#endif

      // perform interval math?
      auto xi = x.template cast<FF>();
      FF s = A.row(i).template cast<FF>().dot(xi);

      // float_t s = A.row(i).dot(x);
      assert(s < .999);  // otherwise this is too close and we are not going to be able to find a close point
      FF b = FF(this->b(i)) - extra_gap_subtract;
      float_t n;
      FF am;
      if (!(s < b)) {  // FML floating point

        am = b * sqrt((FF(1.0) - s * s) / (FF(1.0) - b * b));
#ifdef CERTIFIEDCOSINE_DEBUG_LP_PRINT
        cout << i << " " << b << " " << s << " " << am << " " << x.norm();
#endif

        // .99xx (extra_gap_mul) to try and give a bit of an extra gap to deal with floating point issues
        x -= A.row(i) * upper(s - am * FF(extra_gap_mul));

#ifdef CERTIFIEDCOSINE_DEBUG_LP_PRINT
        cout << " " << x.norm() << " " << loop_times << endl;
#endif

        // check the assert before as the on_sphere_surface might put it back into being violated
        // should be very close to the boundary
        assert(!(A.row(i).template cast<FF>().dot(x.template cast<FF>()) > FF(b)));

        // want this to be back on the circle boundary, would be nice if
        // could just know the value of the norm instead of recomputing it
        // here??
        if (on_sphere_surface) {
          n = x.norm();
          alpha_height /= n;
          x /= n;
        }

        return true;
      }
      return false;
    };

    // given that we are being called the last constraint is likely violated as it should
    // have just been added
    opt_constraint(num_rows - 1);

    while (true) {
#ifdef CERTIFIEDCOSINE_DEBUG
      optimize_loops++;
#endif
      loop_times++;

      // first we have to check all of the rows to see if we are currently violating any of the constraints
      // at each step if we are violating a constraint then we are going to determine what the projection
      // is such that
      violated_constraint = false;
      recheck_constraints = 0;
      for (idx = 0; idx < num_rows; idx++) {
        if (opt_constraint(idx)) {
          // then we were violating this constraint
        located_failed_constraint:
          violated_constraint = true;
          if (idx < 64)  // if this is >= 64 then this just appears to take the lower bits which is annoying
            recheck_constraints |= ((uint64_t)1) << idx;
        }
      }

      if (!on_sphere_surface) {
        FF norm = x.template cast<FF>().norm();
        FF s = c.template cast<FF>().dot(x.template cast<FF>());

        FF pdc, nnorm, keep_o;
        float_t a;

        if (!compareScore(lower(s / norm), proof_distance_cosim)) {
          // the new norm
          pdc = FF(proof_distance_cosim * 1.0075);
          // this is the norm if we were to just add the different from the proof distance.  `s` represents
          // the same pdc, so `sqrt(norm^2 - s^2)` is the orthogonal part to the proof distance
          nnorm = sqrt(norm * norm - s * s + pdc * pdc);
          if (!(nnorm < FF(1.0))) {
            // then the new new norm might be >= 1
            // we have some of the `c` direction contained in `x`, so need to determine how much.
            // `sqrt(1-pdc^2)` is the amount of other direction that we can contain. (1 comes from unit ball)
            // `sqrt((norm^2 - s^2) / norm^2)` is the % of orthogonal stuff contained
            //
            keep_o = sqrt((FF(1.0) - pdc * pdc) / (FF(1.0) - (s * s) / (norm * norm)));
            x = x * lower(keep_o) + c * upper(pdc - keep_o * s / norm);
          } else {
            a = upper(pdc - s);
            x += c * a;
          }
          violated_constraint = true;
        } else if (upper(norm) > 1.0) {
          x /= upper(norm);
        }

        assert(lower(x.template cast<FF>().norm()) <= 1.000005); /* floating point.... */
#ifndef NDEBUG
        FF compare_pd = x.template cast<FF>().dot(c.template cast<FF>()) / x.template cast<FF>().norm();
        assert(compare_pd > proof_distance_cosim);
#endif
      }

    check_after_optimization:
      float_t nl = c.dot(x);
      assert(nl > 0);

      if (!(alpha_height < numeric_limits<float_t>::infinity())) {
        // then this method is failing, return false to reflect that
        return false;
      }
      located_cosim = nl;
      if (!violated_constraint) {
#ifdef CERTIFIEDCOSINE_DEBUG
        assert(check_vector(x));
#endif
        // return true if we found a point that is a counter example
        return compareScore(located_cosim, proof_distance_cosim);
      }

      if (!compareScore(located_cosim, proof_distance_cosim)) {
        // then we are outside of the area that we are interested in, so stop looking
        return false;
      }

      if (on_sphere_surface) {
        if (loop_times == 8) {
          // then we haven't been able to figure out a point within the budget
          return false;
        }
      } else {
        // then we are not on the sphere's surface, so we want to check if the
        // sequence is converging or is a divergent sequence, meaning
        float_t d = (old_x - x).norm();
        if (d > old_x_distances) {
          return false;
        }
        // a sequence converging to zero that should bound the distance between points
        old_x_distances = min(d * 1.5, old_x_distances * .95);
        old_x = x;
      }

      // we are just going to quickly recheck the constraints that were violated last time
      // these are the ones that are most likely to be violated again
      // so hopefully we can more quickly get a pass where nothing is violated
      // only works for the first 64 constraints
      if (__builtin_popcountll(recheck_constraints) >= 2) {
        // if there is only one thing that triggered, then we are likely to get this in the loop again
        // returns zero if the recheck variable is zero
        while ((idx = __builtin_ffsll(recheck_constraints))) {
          recheck_constraints &= recheck_constraints - 1;  // clear the lowest set bit
          opt_constraint(idx - 1);                         // it is the place +1 that is returned
        }
        // these are things that are past the limit of the mask, unlikely to do much
        for (idx = 64; idx < num_rows; idx++) {
          opt_constraint(idx);
        }
      } else if (num_rows <= 64 && on_sphere_surface) {
        // in the case that all the rows would be represented in the recheck_constraints

        // so there is only a single item that failed, so we are just going
        // to check the items before as everything after must have been checked
        int ii = __builtin_ffsll(recheck_constraints);
        assert(ii);               // there must be something that failed, otherwise this is strange
        recheck_constraints = 0;  // zero this out
        violated_constraint = false;
        for (idx = 0; idx < ii - 1; idx++) {  // check the items before
          if (opt_constraint(idx)) {
            // then we found a new constraint that failed, so just restart the optimization loop
            goto located_failed_constraint;
          }
        }
        // then we finished recheck and return
        goto check_after_optimization;
      }
      // if there was a single failure, then it just needs to check the ops that were before
      // but we can't use the bit mask as a definitative count as for values >64
      // it will not be included, so we would need some external count that we check against 1
      // and then try to use the bit mask to determine what item valued.  In that case only
      // we can short cut and just check the earlier items only instead of the whole vector again
      // maybe instead just check that the num_rows <= 64 instead of counting failures?
    }

    assert(false);  // should never get here
    return false;
  }
};

}  // namespace certified_cosine

#endif
