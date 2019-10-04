#ifndef _CERTIFIEDCOSINE_CONSTS
#define _CERTIFIEDCOSINE_CONSTS

#include <limits>

#define CERTIFIEDCOSINE_USE_FLATTABLE
#define CERTIFIEDCOSINE_PREFETCH

#define CERTIFIEDCOSINE_DEBUG

namespace certified_cosine {

template <typename float_t>
struct Consts {
  // constants (should NOT be chanaged)
  static constexpr float_t bestScore = 1;
  static constexpr float_t worseScore = -1;
  static constexpr float_t infWorseScore = -std::numeric_limits<float_t>::infinity();
  static constexpr float_t invalid = std::numeric_limits<float_t>::signaling_NaN();  // error if used

  // constants (can be changed)
  // these can impact the runtime of constructing a proof.
  //
  // these numbers have not been tuned in any meaningful way, however it seems
  // that the dataset and "type" of queries (meaning the distance d(v^*, q))
  // can impact what would be optimal.  I don't have any suggestion for how to
  // tune this values.

  // When searching, there are essentially /two/ vectors that we are having to
  // search for and compare against.  First the query $q$ that we are actually
  // interested in locating the nearest neighbor too, and $x \in S_t$ which is
  // the point that we are looking for to try and fix our certificate.  When $x$ and
  // $q$ are sufficiently close, then it is advantageous to just search for
  // $q$, saving time on the number of inner products that are performed.
  // However, once $x$ is sufficiently far away (we are trying to fill the
  // last gaps in our certificate), then we start using $x$ as the search criteria,
  // however, we still need to compare against $q$ in the case that we find a
  // better neighbor, thus this requires two inner products.
  static constexpr float_t different_proof_distance = .85;
  static constexpr float_t percent_proof_distance = .73;

  // When a new $x \in S_t$ we may want to reconsider points that we have
  // already observed but have not fully processed as they might be useful for
  // constructing a certificate.  However, reconsidering points is expensive,
  // so this controls how frequently we allow that to happen.
  static constexpr float_t different_score_step_distance = .92;

  // When a new point is expanded, we cap the size of the queue otherwise
  // essentially all $v_i \in V$ would end up in the queue.
  //
  // The search will terminate if the queue is exhausted, though that does not
  // necessarily mean that a certificate has been constructed in that case.
  static constexpr int queue_size = 20;
};

template <typename float_t>
inline static bool compareScore(float_t a, float_t b) {
  // Determine what the ordering. Return true if a is /better/ than b.  Better
  // scores are those that are closer to the target, and thus /larger/ in
  // value given that the comparison is performed using cosine similarity.
  return a > b;
}

}  // namespace certified_cosine

#endif
