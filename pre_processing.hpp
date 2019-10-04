#ifndef _CERTIFIEDCOSINE_PRE_PROCESSING
#define _CERTIFIEDCOSINE_PRE_PROCESSING

#include <Eigen/Dense>
#include "constants.hpp"
#include "storage.hpp"

namespace certified_cosine {

template <typename float_t>
using PMatrix = Eigen::Ref<Eigen::Matrix<float_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>;

template <typename float_t>
void exact_neighbors(const PMatrix<float_t> &matrix, dynamic_storage<float_t> &storage, uint num_neighbors = 10);

template <typename float_t>
void reverse_edges(dynamic_storage<float_t> &storage);

template <typename float_t>
void build_all_edges(dynamic_storage<float_t> &storage);

template <typename float_t>
void starting_approximations(const PMatrix<float_t> &matrix, dynamic_storage<float_t> &storage, uint starting_points);

template <typename float_t>
void shuffle_all_edges(const PMatrix<float_t> &matrix, dynamic_storage<float_t> &storage, int num_incoming_edges);

// redo the exact neighbors
template <typename float_t>
void make_smaller(const dynamic_storage<float_t> &input_storage, dynamic_storage<float_t> &output_storage,
                  uint new_num_neighbors);

/**
 * Do everything required
 */
template <typename float_t>
void preprocess(const PMatrix<float_t> &matrix, dynamic_storage<float_t> &storage, uint num_neighbors = 10,
                uint starting_points = 1 << 16) {
  exact_neighbors(matrix, storage, num_neighbors);
  reverse_edges(storage);
  //#ifndef CERTIFIEDCOSINE_WEIGHT_DIST
  shuffle_all_edges(matrix, storage, num_neighbors / 3);
  //#endif
  build_all_edges(storage);
  starting_approximations(matrix, storage, starting_points);
}

template <typename float_t>
void make_smaller_all(const PMatrix<float_t> &matrix, const dynamic_storage<float_t> &input_storage,
                      dynamic_storage<float_t> &output_storage, uint new_num_neighbors = 10,
                      uint starting_points = 1 << 16) {
  make_smaller(input_storage, output_storage, new_num_neighbors);
  reverse_edges(output_storage);
  shuffle_all_edges(matrix, output_storage, new_num_neighbors / 3);
  build_all_edges(output_storage);
  starting_approximations(matrix, output_storage, starting_points);
}

}  // namespace certified_cosine

#endif
