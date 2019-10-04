#ifndef _CERTIFIEDCOSINE_HASH_LOOKUP
#define _CERTIFIEDCOSINE_HASH_LOOKUP

#include <math.h>

namespace certified_cosine {

// compute a fast hash of a vector by checking the sign bits of the first
// dimensions assuming that the vectors are from a /dense/ embedding, where no
// special meaning is assigned to the different dimensions, then this is fine
// without having to first perform a random rotation of the vector.
template <typename vec_t>
uint32_t compute_signature(const vec_t &vec) {
  uint32_t ret = 0;
  const uint l = min((uint)vec.size(), (uint)32);
  for (uint i = 0; i < l; i++) {
    // if <0 then it gets a 1.
    ret |= signbit(vec(i)) << i;
  }
  return ret;
}

}  // namespace certified_cosine

#endif
