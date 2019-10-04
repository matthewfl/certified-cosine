#include "vector_stats.hpp"

#include <math.h>
#include <algorithm>
#include <iostream>

namespace certified_cosine {

using namespace std;

template <typename float_t>
std::vector<float_t> summarize_single_distance_proof(const dynamic_storage<float_t> &storage) {
  vector<float_t> dists(storage.size());
  for (size_t i = 0; i < storage.size(); i++) {
    dists[i] = sqrt(1 + storage.get_vertex(i).proof_distance(&storage)) / sqrt(2);
  }
  sort(dists.begin(), dists.end());

  vector<float_t> ret(100);
  for (int i = 0; i < 100; i++) {
    ret[i] = dists[dists.size() - 1 - i * dists.size() / 100];
  }
  return ret;
}

template std::vector<float> summarize_single_distance_proof(const dynamic_storage<float> &storage);
template std::vector<double> summarize_single_distance_proof(const dynamic_storage<double> &storage);

template <typename float_t>
std::vector<int> summarize_num_neighbors(const dynamic_storage<float_t> &storage) {
  vector<int> neighbor(storage.size());
  for (size_t i = 0; i < storage.size(); i++) {
    neighbor[i] = storage.get_vertex(i).size(&storage);
  }
  sort(neighbor.begin(), neighbor.end());
  vector<int> ret(100);
  for (int i = 0; i < 100; i++) {
    ret[i] = neighbor[neighbor.size() - 1 - i * neighbor.size() / 100];
  }
  return ret;
}

template std::vector<int> summarize_num_neighbors(const dynamic_storage<float> &storage);
template std::vector<int> summarize_num_neighbors(const dynamic_storage<double> &storage);

template <typename float_t>
void print_summarization(const dynamic_storage<float_t> &storage) {
  vector<float_t> pd = summarize_single_distance_proof(storage);
  vector<int> nc = summarize_num_neighbors(storage);

  cout << "Single proof distance (cosine > than this value)\n"
          "     20%: "
       << pd[80]
       << "\n"
          "     40%: "
       << pd[60]
       << "\n"
          "     60%: "
       << pd[40]
       << "\n"
          "     80%: "
       << pd[20]
       << "\n"
          "    100%: "
       << pd[0]
       << "\n"
          "\n"
          "Num neighbors \n"
          "      0%: "
       << nc[0]
       << " (max)\n"
          "     20%: "
       << nc[20]
       << "\n"
          "     40%: "
       << nc[40]
       << "\n"
          "     60%: "
       << nc[60]
       << "\n"
          "     80%: "
       << nc[80] << "\n";
}

template void print_summarization(const dynamic_storage<float> &storage);
template void print_summarization(const dynamic_storage<double> &storage);

}  // namespace certified_cosine
