#ifndef _CERTIFIEDCOSINE_VECTOR_STATS
#define _CERTIFIEDCOSINE_VECTOR_STATS

#include <vector>

#include "storage.hpp"

namespace certified_cosine {

/**
 * determine what distances can be proven for a point using only that point.
 */
template <typename float_t>
std::vector<float_t> summarize_single_distance_proof(const dynamic_storage<float_t> &storage);

template <typename float_t>
std::vector<int> summarize_num_neighbors(const dynamic_storage<float_t> &storage);

template <typename float_t>
void print_summarization(const dynamic_storage<float_t> &storage);

}  // namespace certified_cosine

#endif
