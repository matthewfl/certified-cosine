/**
 * Written By Matthew Francis-Landau (2019)
 *
 * Main lookup procedure for certified cosine
 */

#ifndef _CERTIFIEDCOSINE_LOOKUP
#define _CERTIFIEDCOSINE_LOOKUP

#include <assert.h>
#include <tuple>

// #ifdef DEBUG_LOOKUP
// # include <iostream>
// #endif

#include <Eigen/Dense>

#include "constants.hpp"
#include "lp_project.hpp"
#include "lp_simplex.hpp"
#include "utils.hpp"

#include <boost/numeric/interval.hpp>
#include <boost/numeric/interval/transc.hpp>
#include "feroundingmode.h"  // include this header before boost interval math library (avoid using glibc through a indirect library call)

#include "vector_signature.hpp"

namespace certified_cosine {

using namespace std;

template <typename storage_t,
          typename matrix_t,  // eigen matrix type, should probably be an Eigen::Ref
          typename vector_t = Eigen::Matrix<typename matrix_t::Scalar, 1, matrix_t::ColsAtCompileTime>>
class LookupCertifiedCosine {
 public:
  typedef typename matrix_t::Scalar float_t;

  typedef matrix_t MatD;
  typedef vector_t VecD;

 private:
#ifdef CERTIFIEDCOSINE_DEBUG_ACCESS
 public:
#endif

  // references to the stored matrix and the KNNG used for search
  const matrix_t matrix;
  const storage_t *const storage;

  // row 0 is the origional target that we are comparing against
  // row 1 is the one that changes and we are looking for
  Eigen::Matrix<float_t, 2, matrix_t::ColsAtCompileTime, Eigen::RowMajor> target;
  // the distance between the target and $x \in S_t$ that we have located
  float_t located_cosim = 1;

  // the last x when we increased the score step
  VecD target_last_point;

  // this tracks the current score step and is compared against the
  // vertexHashedInfo.score_step
  uint score_step = 0;

  // this is the current size of $N_{\hat{v}}(q)$ that we are trying to
  // construct a certificate for
  float_t min_proof_distance = Consts<float_t>::worseScore;

  // the solvers used for tracking $S_t$ and eventually identifying that $S_t =
  // \emptyset$ indicating that a certificate is successfully constructed.
  LPSolverProject<float_t, matrix_t::ColsAtCompileTime> lp_project;
  LPSolverSimplex<float_t> lp_simplex;
  int lp_simplex_inited = 0;

  struct vertexHashedInfo {
    float_t score;                   // score compared to the looking_target
    /*const*/ float_t target_score;  // score compared to the target

    // reference to lookup information about this vertex
    // the id of the vertex is stored in the ref
    typename storage_t::vertex ref;

    int16_t processed = 0;
    int16_t processed_high = 0;

    // this tracks the last time that we compared against this vector.  In the
    // case that we are constructing a certificate, it can be beneficial to
    // reconsider vectors that we have already expanded.
    uint score_step : 15;

    uint added_to_proof : 1;

    // for the hash table
    inline int get_key() const { return ref.get_id(); }
    inline int get_id() const { return ref.get_id(); }

    vertexHashedInfo() : target_score(Consts<float_t>::invalid), score_step(0), added_to_proof(0) {}

    vertexHashedInfo(int id, float_t score, float_t target_score, int score_step, const typename storage_t::vertex &ref)
        : target_score(target_score), score_step(score_step), ref(ref), added_to_proof(0) {
      assert(id == ref.get_id());
    }

    inline bool operator==(const vertexHashedInfo &o) const { return o.get_id() == get_id(); }
  };

  struct vertexQueueInfo {
    float_t score;
    int id;
    // for the IntervalHeap
    inline float_t get_value() const {
      assert(id != -1);
      return score;
    }
    vertexQueueInfo(int id, float_t score) : score(score), id(id) {}
    vertexQueueInfo() : score(Consts<float_t>::invalid), id(-1) {}
    inline bool operator==(const vertexQueueInfo &o) const { return o.id == id; }
  };

  // tracks which vertices have been expanded.  The hash table uses less memory
  // at the expense of some speed.  The flat table will initialized to the size
  // of the number of vectors in $\mathcal{V}$ and thus require more space.
#ifdef CERTIFIEDCOSINE_USE_FLATTABLE
  FlatTable<vertexHashedInfo> expanded;
#else
  CuckooHashTable<vertexHashedInfo> expanded;
#endif

  IntervalHeap<vertexQueueInfo> queue;
  IntervalHeap<vertexQueueInfo> queue2;

#ifndef CERTIFIEDCOSINE_USE_FLATTABLE
  // the LRU saves time on avoidable hash table lookups
  LRUContains<int, 1024> lru;
#endif

  void queue_insert(vertexQueueInfo &q, bool &queued) {
    if (queue.size() < Consts<float_t>::queue_size) {
      queue.insert(q);
      queued = true;
    } else if (compareScore(q.score, queue.min().score)) {
      queue.remove_min();
      queue.insert(q);
      queued = true;
    }
  }

  template <bool looking_not_target = true>
  inline void expanded_insert(const vertexHashedInfo &o) {
    if (looking_not_target) {
      expanded.insert(o, [this](const vertexHashedInfo &d) {
        // this allows us to drop data from the hash table that
        // is "expired" which can save time during the insert
        // procedure
        return !(d.score_step + 1 >= score_step || d.added_to_proof || d.processed_high == d.ref.size(storage));
      });
    } else {
      // the check would not work in this case, so don't bother with the check
      expanded.insert(o);
    }
  }

  template <bool looking_not_target, typename policy_t>
  inline bool expand_point(int id, bool &queued, policy_t &policy) {
#ifndef CERTIFIEDCOSINE_USE_FLATTABLE
    if (lru(id)) return false;
#endif
    vertexHashedInfo *i = expanded.lookup(id);
    if (i == nullptr) {
      if (looking_not_target) {
        // this computes the inner product with the origional target and
        Eigen::Matrix<float_t, 2, 1> scores = matrix.row(id) * target.transpose();

        if (policy.expand(id, scores(0))) return true;
        auto sref = storage->get_vertex(id);
        vertexHashedInfo h(id, scores(1), scores(0), score_step, sref);
        expanded_insert<looking_not_target>(h);

        vertexQueueInfo q(id, scores(1));
        queue_insert(q, queued);
      } else {
        // there is only a single point here, so we don't have to double the
        // number of inner products that we are performing
        float_t score = matrix.row(id).dot(target.row(0));

        if (policy.expand(id, score)) return true;
        auto sref = storage->get_vertex(id);
        vertexHashedInfo h(id, score, score, score_step, sref);
        expanded_insert<looking_not_target>(h);

        vertexQueueInfo q(id, score);
        queue_insert(q, queued);
      }
    } else if (looking_not_target) {
      expand_point(i, queued);
    }
    return false;
  }

  inline void expand_point(vertexHashedInfo *i, bool &queued) {
    if (i->score_step == score_step) return;  // already evaluated this for this value
    if (i->added_to_proof && i->processed == i->ref.size(storage)) return;
    float_t xscore = matrix.row(i->get_id()).dot(target.row(1));
    if (xscore > i->score) i->processed = 0;  // reset the number processed so that we redo this item?
    i->score = xscore;
    i->score_step = score_step;
    vertexQueueInfo q(i->get_id(), xscore);
    queue_insert(q, queued);
  }

  template <typename policy_t>
  bool add_to_proof(vertexHashedInfo *h, policy_t &policy) {
    // take the newly processed vertex `h`, and add it to the proof it can make
    // some change this will identify a new point $x \in S_t$ (if it exists).
    // If the new point is sufficiently unique, then we can mark it as a new
    // score_step which will allow us to reevaluate previously checked vectors

    // this is the distance that we have to cover.  This represents the size of
    // the neighborhood around the query.  Using the policy allows us to switch
    // between the 1-best or try and prove the top k-best etc.
    min_proof_distance = policy.proof_distance();

    // there is no point in trying to construct in a certificate in this case.
    // To construct a certificate for the worseScore value that would require
    // searching the entire space (achieve no speedup).  It is possible to use
    // this value to indicate that not enough of the space has been observed yet
    if (min_proof_distance <= Consts<float_t>::worseScore) return false;
    const float_t proof_dist = h->ref.proof_distance(storage);

    {
      // check if a single point certificate can be constructed
      using namespace boost::numeric;
      using namespace interval_lib;
      using namespace compare::certain;

      typedef interval<float_t, policies<rounded_transc_std<float_t>, checking_strict<float_t>>> FF;

      if (h->target_score >= .999 || min_proof_distance >= .984 ||
          acos(FF(min_proof_distance)) + acos(FF(h->target_score)) < acos(FF(proof_dist)))
        return true;
    }

    // compute the distance from the point that we are interested in
    h->score = matrix.row(h->get_id()).dot(target.row(1));

    // then this isn't close enough to the target to attempt to add it to the
    // proof at this time
    if (!compareScore(h->score, proof_dist) &&
        located_cosim >
            min_proof_distance  // the proof distance has changed so we might still be unable to find a new point?
    ) {
      return false;
    }

  add_point:;
    assert(!h->added_to_proof);

    // load this new point into the projection value
    h->added_to_proof = true;
    assert(h == expanded.lookup(h->get_id()));

    lp_project.resize(lp_project.num_rows + 2);
    lp_project.A.row(lp_project.num_rows) = matrix.row(h->get_id());

    lp_project.d(lp_project.num_rows) = h->target_score;

    lp_project.b(lp_project.num_rows++) = proof_dist;

  project_optimize:;

    lp_project.proof_distance_cosim = min_proof_distance;

    // this returns true in the case it locates a point $x \in S_t$ which is
    // contained on the surface of the sphere.  We can directly use this point
    // without having to perform additional relaxations with the simplex linear
    // program.  However, failure to find a point using this method does not
    // mean that $S_t$ is empty.
    if (lp_project.optimize()) {
      target.row(1) = lp_project.x;

#ifdef CERTIFIEDCOSINE_DEBUG
      assert(lp_project.check_vector(target.row(1)));
#endif

      goto new_point;
    } else if (lp_project.num_rows == 1) {
      // in the case that there is only a single constraint, then this point is
      // the closest point $x$ to $q$ and thus we can use this to directly check
      // if it would be contained in $S_t$.

      if (lp_project.located_cosim < min_proof_distance) return true;
    }

  simplex_optimize:;
    // In this case, the project method was unable to locate a $x \in S_t$ on
    // its own.  So we combine the result of the projection method with a linear
    // programming simplex solver via two convex relaxations that we can solve
    // which allows us to find $x \in S_t$

    // this finds a point in $conv(S_t)$.  In the case it can't find a point,
    // this returns false which means that $conv(S_t) = \emptyset$, thus $S_t =
    // \emptyset$ as well and a certificate is constructed.
    if (!lp_project.optimize(false)) {
      return true;
    }

    if (lp_simplex_inited == 0) {
    simplex_reset:;
      lp_simplex.load_tableau(lp_project.A.topRows(lp_project.num_rows), lp_project.b.topRows(lp_project.num_rows),
                              lp_project.c);
      lp_simplex_inited = lp_project.num_rows;
    } else {
      for (; lp_simplex_inited < lp_project.num_rows; lp_simplex_inited++) {
        lp_simplex.add_constraint(lp_project.A.row(lp_simplex_inited), lp_project.b(lp_simplex_inited),
                                  lp_project.d(lp_simplex_inited));
      }
      // the dual simplex seems to have floating point instability issues
      // sometimes, so perform a tableau reset in these cases.
      if (!lp_simplex.run_dual_simplex()) goto simplex_reset;
    }

    lp_simplex.run_simplex();

    if (!lp_simplex.is_constraint_tight()) {
      // then we do not have c^T x = 1.  given that we are not restricted to the
      // unit ball, this means that there must not be any point that is better
      // than c^T x.  If the objective value is lower than the distance that are
      // trying to prove, then we are done

      if (lp_simplex.objective() < min_proof_distance) {
        return true;
      }

      // if we are still not proven something, if we are inside of the unit
      // ball, then we can't prove emptyness, so we are just going to return and
      // continue looking.  If we are outside of the unit ball, then we can
      // still use this point to help with finding x \in S_t

      if (lp_simplex.get_x_vec().norm() < 1) {
        // we can't use this point to help, and we can't prove it is empty
        return false;
      }
    }

    // we are going to define the newly located point as the point between the
    // simplex method and the projection method given that this is convex space
    // (ignoring ||x||=1) then that means all the points between must also
    // respect the constraints and then we can just look for the point where
    // ||x||=1 along the line
    {
      auto simplex_x = lp_simplex.get_x_vec().transpose().eval();

      auto diff = simplex_x - lp_project.x;
      float_t proj_norm = lp_project.x.norm();

      float_t diff_norm = diff.norm();

      float_t angle = diff.dot(lp_project.x) / (diff_norm * proj_norm);

      float_t l = sqrt((1 - proj_norm * proj_norm) / (1 - angle * angle));
      target.row(1) = lp_project.x + diff / diff_norm * l;
      lp_project.x = target.row(1);
    }

  new_point:;
    // there is a new point, so we are going to want to reshuffle the queue to
    // see if there is a new item that we can use to prove this point.
    //
    // this is one of the interesting things that makes this different from
    // previous nearest neighbor methods, however, it is /very/ expensive as it
    // is essentially starting a new search.
    //
    // As such, we are going to try and avoid increasing the score step as much
    // as possible.  We /might/ change the searching target without increasing
    // the score step which can impact what the order of the queue should be,
    // but we are just going to deal with that little difference in error,
    // letting new points get sorted according to the new distance

    located_cosim = target.row(1).dot(target.row(0));

    if (score_step == 0) {
      // check if this point is sufficiently far enough away that we want to
      // switch to the slower method where we compute the inner product between
      // two vectors rather than just 1.  This should save time during the
      // earlier stages of lookup as we are not going to be doing double the
      // number of inner products.
      if (located_cosim > Consts<float_t>::different_proof_distance ||
          (1 - located_cosim) / (1 - min_proof_distance) < Consts<float_t>::percent_proof_distance) {
        return false;
      }

      target_last_point = target.row(1);
    } else {
      // the inner product between the last time we increased the score step and
      // where we are currently.
      float_t d = target_last_point.dot(target.row(1));

      if (d > Consts<float_t>::different_score_step_distance) return false;

      target_last_point = target.row(1);
    }

    queue2.clear();
#ifndef CERTIFIEDCOSINE_USE_FLATTABLE
    lru.clear();
#endif
    swap(queue, queue2);

    score_step++;  // track that we have moved to the next step in trying to construct a certificate

    // this is probably not worth reprocessing the item in this case, there
    // shouldn't be anything closer to the target that we are trying to find, as
    // we have located a close point to the target that is not covered.  If that
    // close point had something that was covered using this item then we
    // probably would have been proving something using that instead.

    bool queued;
    expand_point(h, queued);

    assert(h == expanded.lookup(h->get_id()));
    h = nullptr;  // ensure we don't use this

    // retry the O(1) seed hash lookup
    if (expand_point<true, policy_t>(lookup_init_point(), queued, policy)) return true;

    int id = -1;
    while (queue2.size()) {
      vertexQueueInfo q = queue2.max();
      queue2.remove_max();
      // insert from queue2 into queue
      if (expand_point<true, policy_t>(q.id, queued, policy)) return true;

      if (queue.max().id != id) {
        q = queue.max();
        id = q.id;
        vertexHashedInfo *i = expanded.lookup(id);
        if (compareScore(q.score, i->ref.proof_distance(storage))) {
          // then this item can work as a proof so we want to process it
          i->processed = i->processed_high;
          break;
        }
      }
    }

    return false;
  }

  template <bool looking_not_target, typename policy_t>
  bool lookup_using_queue(policy_t &policy) {
    bool queued = false;
    vertexQueueInfo q;
    vertexHashedInfo h;
    int number_edges;
    int n;

    while (queue.size()) {
      q = queue.max();

      h = *expanded.lookup(q.id);
      number_edges = h.ref.size(storage);

      // shortcut this case for the single point proof
      float_t scratch = policy.proof_distance();
      scratch = 2 * scratch * scratch;  // scratch might be -inf in the case that we haven't yet seen enough points
      if (scratch > h.ref.proof_distance(storage) + 1 && scratch < 10) {
        // if the proof distance is small enough then we might be able to do the
        // single point proof if the proof distance is large enough

        // but we also have to consider the distance of the point from the target
        // in which case we need to sum the two angles
        // use interval math as we have to be sure in this case
        using namespace boost::numeric;
        using namespace interval_lib;
        using namespace compare::certain;

        typedef interval<float_t, policies<rounded_transc_std<float_t>, checking_strict<float_t>>> FF;

        FF pd = policy.proof_distance();
        if (h.target_score >= .998 /* if the score is >1, then the acos line can fail...floating point issues */ ||
            (acos(pd) + acos(FF(h.target_score)) < acos(FF(h.ref.proof_distance(storage)))))
          goto do_single_point_proof;
      }

    expand_point:
      queued = false;
      auto neighbor_opaque = h.ref.neighbor_opaque(storage);
      while (h.processed < number_edges) {
        n = h.ref.neighbor(storage, neighbor_opaque, h.processed);

        h.processed++;

#ifdef CERTIFIEDCOSINE_PREFETCH
        // prefetch the following entry so that on the next pass through the loop it will be ready
        if (h.processed < number_edges) {
          int n2 = h.ref.neighbor(storage, neighbor_opaque, h.processed);
          expanded.prefetch(n2);
          __builtin_prefetch(matrix.row(n2).data(), 0, 0);
        }
#endif

        if (expand_point<looking_not_target, policy_t>(n, queued, policy))  // then the policy returned that we are done
          return true;

        // if something was added to the queue, then it might be better than the current vector
        if (queued) {
          // check if we are still the top element in the queue
          if (queue.max().id != h.get_id()) {
            goto stash_point;
          }
          queued = false;
        }
      }
      // delete ourselves from the queue as done processing
      assert(queue.max().id == h.get_id());
      queue.remove_max();

    stash_point:;
      // take the point and put it back into the queue.  But it will now it
      // might have a lower score given that we are later in the processed items
      // there might also be something that is better than what we were
      // processing.
      {
        if (h.processed_high < h.processed) h.processed_high = h.processed;

        vertexHashedInfo *hptr = expanded.insert(h);
        if (!hptr->added_to_proof && h.processed >= storage->proof_distance_size()) {
          // try adding this point to the proof.  We have processed enough of
          // its vertices such that the proof distance could be useful
          if (add_to_proof(hptr, policy)) return true;

          // then we want to run version which is going to compare against both
          // vectors $x \in S_t$ and $q$ as they are now sufficiently different.
          if (!looking_not_target && score_step != 0) {
            return false;
          }
        }
      }

    }  // end while(queue.size())

    return false;

  do_single_point_proof:
    // this point is sufficient to do the single point proof so we are just
    // going to run until we have looked at enough of the neighbors, and then
    // return.
    auto neighbor_opaque = h.ref.neighbor_opaque(storage);
    while (h.processed < storage->proof_distance_size()) {
      int n = h.ref.neighbor(storage, neighbor_opaque, h.processed);
      float_t score = matrix.row(n).dot(target.row(0));
      if (policy.expand(n, score)) return true;
      h.processed++;
    }
    return true;
  }

  int lookup_init_point() {
    // return an initial point in O(1) time based off the "signature" of the
    // point that we are currently looking for

    uint32_t signature = compute_signature(target.row(1));
    return storage->get_starting(signature);
  }

  template <typename policy_t>
  bool lookup_outer(policy_t &policy) {
    // we first start try to avoid doing extra work when we are in the simple case
    if (lookup_using_queue<false, policy_t>(policy)) return true;

    // once we have started locating new areas that we are interested in, we
    // have to compute between the two different vectors, so we use this
    // alternate version of the function which will perform extra work
    return lookup_using_queue<true, policy_t>(policy);
  }

 public:
  LookupCertifiedCosine(const matrix_t &matrix, const storage_t *storage)
      : matrix(matrix),
        storage(storage)
#ifdef CERTIFIEDCOSINE_USE_FLATTABLE
        ,
        expanded(matrix.rows())
#endif
  {
    lp_project.init(matrix.cols());
  }

  LookupCertifiedCosine(const LookupCertifiedCosine &other) : LookupCertifiedCosine(other.matrix, other.storage) {}

  template <typename policy_t>
  bool lookup(const VecD &x, policy_t &policy) {
    // reset queues
    queue.clear();
    queue2.clear();
    expanded.soft_clear();
#ifndef CERTIFIEDCOSINE_USE_FLATTABLE
    lru.clear();
#endif
    score_step = 0;

    // reset proof
    lp_project.reset();
    min_proof_distance = Consts<float_t>::worseScore;
    lp_simplex_inited = 0;
    located_cosim = 1;

    // load the data
    target = x.template replicate<2, 1>();

    lp_project.x = x;
    lp_project.c = x;

    // get the seed point for the queues
    int start = lookup_init_point();
    bool queued;
    if (expand_point<false, policy_t>(start, queued, policy))
      return true;  // then the policy has terminated after one point

    // run the lookup loop
    return lookup_outer(policy);
  }
};

}  // namespace certified_cosine

#endif
