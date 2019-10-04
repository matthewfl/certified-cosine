#include <omp.h>
#include <stdio.h>
#include <algorithm>
#include <array>
#include <vector>

#include "constants.hpp"
#include "pre_processing.hpp"
#include "utils.hpp"
#include "vector_signature.hpp"

// todo: tune this parameter?
#define BATCH_SIZE 10

using namespace certified_cosine;
using namespace std;

#ifndef CERTIFIEDCOSINE_USE_PARALLEL
#define omp_get_thread_num() 0
#endif

namespace certified_cosine {

template <typename float_t>
void exact_neighbors(const PMatrix<float_t> &matrix, dynamic_storage<float_t> &storage, uint num_neighbors) {
  time_t start = time(NULL);

  uint processed_vertexes = 0;
  const uint num_vertices = matrix.rows();

  if (num_neighbors > num_vertices) num_neighbors = num_vertices;

  storage.set_num_vertexes(num_vertices, num_neighbors);

  typedef typename dynamic_storage<float_t>::edge edge_p;
  struct edge : edge_p {
    edge(int d, float_t s) : edge_p(d, s) {}
    edge() : edge_p(-1, Consts<float_t>::worseScore) {}
    // sort the smallest values first so that we can remove them easily from the heap
    bool operator<(const edge &o) { return this->score > o.score; }
  };

#ifdef CERTIFIEDCOSINE_USE_PARALLEL
#pragma omp parallel
#endif
  {
    array<float_t, BATCH_SIZE> worst_scores;
    array<vector<edge>, BATCH_SIZE> edges;

    const int omp_id = omp_get_thread_num();
    uint8_t printcnt = 0;

#ifdef CERTIFIEDCOSINE_USE_PARALLEL
#pragma omp for schedule(dynamic, 50)
#endif
    for (int i = 0; i < num_vertices; i += BATCH_SIZE) {
      // process the first j just putting these into the heap so that it is the right size
      for (int k = 0; k < BATCH_SIZE; k++) {
        if (i + k >= num_vertices) continue;
        edges[k].resize(num_neighbors);
        for (int j = 0; j < num_neighbors; j++) {
          if ((i + k) != j) {
            float_t scr = matrix.row(i + k).dot(matrix.row(j));
            edges[k][j] = edge(j, scr);
          }
        }
        // make this into a heap
        make_heap(edges[k].begin(), edges[k].end());
        worst_scores[k] = edges[k][0].score;
      }

      // process the remaining items
      for (int j = num_neighbors; j < num_vertices; j += 4) {
        for (int k = 0; k < BATCH_SIZE; k++) {
          if (i + k >= num_vertices) continue;
          for (int l = 0; l < 4; l++) {
            if (l + j >= num_vertices) continue;
            float_t scr = matrix.row(i + k).dot(matrix.row(l + j));
            if (worst_scores[k] < scr && (i + k) != (l + j)) {
              pop_heap(edges[k].begin(), edges[k].end());
              edges[k].back() = edge(l + j, scr);
              push_heap(edges[k].begin(), edges[k].end());
              worst_scores[k] = edges[k][0].score;

              // I suppose that it is possible that insert sort might be
              // better assuming that we are usually changing one of the lower
              // valued items instead of the top valued.  So in this case we
              // are do log operations each time that there is an insert

              // auto &oe = vertexes[i + k].outgoing_edges;
              // oe.back() = edge(l + j, scr);
              // simpleSort(oe);
              // worst_scores[k] = oe.back().score;
            }
          }
        }
      }

      // now we have to save this into the storage
      for (int k = 0; k < BATCH_SIZE; k++) {
        if (k + i >= num_vertices) continue;
        sort_heap(edges[k].begin(), edges[k].end());
        auto ref = storage.get_vertex(k + i);
        auto &outgoing = ref.outgoing_edges(&storage);
        outgoing.resize(num_neighbors);
        for (int j = 0; j < num_neighbors; j++) {
          outgoing[j] = edges[k][j];
        }
        float_t pd = outgoing[num_neighbors - 1].score;
        if (pd >= 1) pd = 1;
        ref.set_proof_distance(&storage, num_neighbors - 1, pd);
      }

      processed_vertexes += BATCH_SIZE;  // no locking as we are just getting an approximation
      if (omp_id == 0 && printcnt++ % 50 == 0) {
        // print our a progress of how far this has gotten
        int secs = difftime(time(NULL), start);
        float done = ((float)processed_vertexes) / num_vertices;
        printf("       \rProcessing exact first part: %.2f%%, remaining mins: %i/%i", done * 100,
               (int)(secs / done * (1 - done) / 60), (int)(secs / done / 60));
        fflush(stdout);
      }
    }

    if (omp_id == 0) printf("\n");
  }
}

template void exact_neighbors(const PMatrix<float> &matrix, dynamic_storage<float> &storage, uint num_neighbors);
template void exact_neighbors(const PMatrix<double> &matrix, dynamic_storage<double> &storage, uint num_neighbors);

template <typename float_t>
void reverse_edges(dynamic_storage<float_t> &storage) {
#ifdef CERTIFIEDCOSINE_USE_PARALLEL
  // make this a global array, so that this can be reused between different stages?
  // can make the add method external also so that this can be reused.
  array<SpinLock, 10000> locks;
#endif

  typedef typename dynamic_storage<float_t>::edge edge;

  time_t start = time(NULL);

  uint processed_vertexes = 0;

#ifdef CERTIFIEDCOSINE_USE_PARALLEL
#pragma omp parallel
#endif
  {
    // once we are done proccessing the first set of the edges, then we should move onto ensuring that there are
    // linked in both directions.  This is going to mean taking a lock on the edges when we have to add something
    // to the remote edge, and then having that as some incoming edge.

    const int omp_id = omp_get_thread_num();
    uint8_t printcnt = 0;

#ifdef CERTIFIEDCOSINE_USE_PARALLEL
#pragma omp for schedule(dynamic, 100)
#endif
    for (int i = 0; i < storage.size(); i++) {
      auto vx = storage.get_vertex(i);
      for (auto e : vx.outgoing_edges(&storage)) {
#ifdef CERTIFIEDCOSINE_USE_PARALLEL
        locks[e.id % locks.size()].lock();
#endif
        auto ovx = storage.get_vertex(e.id);
        ovx.incoming_edges(&storage).push_back(edge(i, e.score));
#ifdef CERTIFIEDCOSINE_USE_PARALLEL
        locks[e.id % locks.size()].unlock();
#endif
      }

      processed_vertexes++;
      if (omp_id == 0 && printcnt++ % 50 == 0) {
        // print our a progress of how far this has gotten
        int secs = difftime(time(NULL), start);
        float done = ((float)processed_vertexes) / storage.size();
        printf("       \rProcessing reverse part: %.2f%%, remaining mins: %i/%i", done * 100,
               (int)(secs / done * (1 - done) / 60), (int)(secs / done / 60));
        fflush(stdout);
      }
    }

    if (omp_id == 0) printf("\n");
  }

  // #ifdef CERTIFIEDCOSINE_WEIGHT_DIST
  // #ifdef USE_PARALLEL
  // #pragma omp for schedule(dynamic, 100)
  // #endif
  //   for(int i = 0; i < storage.size(); i++) {
  //     auto vx = storage.get_vertex(i);
  //     auto &iedges = vx.incoming_edges(&storage);
  //     // put the largest elements first in the list, as those are the ones that we want to explore first
  //     std::sort(iedges.begin(), iedges.end(), [](const auto &a, const auto &b) { return a.score > b.score; });
  //   }

  // #endif
}

template void reverse_edges(dynamic_storage<float> &storage);
template void reverse_edges(dynamic_storage<double> &storage);

template <typename float_t>
void build_all_edges(dynamic_storage<float_t> &storage) {
#ifdef CERTIFIEDCOSINE_USE_PARALLEL
#pragma omp parallel for schedule(dynamic, 100)
#endif
  for (int i = 0; i < storage.size(); i++) {
    auto vx = storage.get_vertex(i);
    vx.build_all_edges(&storage);
  }
}

template void build_all_edges(dynamic_storage<float> &);
template void build_all_edges(dynamic_storage<double> &);

template <typename float_t>
void starting_approximations(const PMatrix<float_t> &matrix, dynamic_storage<float_t> &storage, uint starting_points) {
  // compute vertices which would serve as good starting points
  assert(__builtin_popcount(starting_points) == 1);  // check is some power of two

  auto &startings = storage.starting_arr();
  startings.resize(starting_points);
  // set a default value
  fill(startings.begin(), startings.end(), -1);

  const uint mask = starting_points - 1;

#ifdef CERTIFIEDCOSINE_USE_PARALLEL
#pragma omp for schedule(dynamic, 100)
#endif
  for (int i = 0; i < matrix.rows(); i++) {
    // compute the signatures of all the vectors and assign them to the starting
    // array we do not care about which "valid" vector is assigned to a space at
    // this time, just that we find something, so we do not lock or try to avoid other threads
    startings[compute_signature(matrix.row(i)) & mask] = i;
  }

  int last_val, i;
  for (i = 0; i < starting_points; i++) {
    if (startings[i] != -1) {
      for (int j = 0; j < i; j++) startings[j] = startings[i];
      last_val = startings[i];
      break;
    }
  }
  for (; i < starting_points; i++) {
    if (startings[i] == -1)
      startings[i] = last_val;
    else
      last_val = startings[i];
  }
}

template void starting_approximations(const PMatrix<float> &matrix, dynamic_storage<float> &storage, uint);
template void starting_approximations(const PMatrix<double> &matrix, dynamic_storage<double> &storage, uint);

template <typename float_t>
void super_nodes(const PMatrix<float_t> &matrix, dynamic_storage<float_t> &storage) {
  // when running with /real/ data, like word vectors, there are some vertices
  // which have a lot more incoming edges than outgoing
  //
  // for example with real data, there may be some nodes which have 50k incoming edges
  // it would be better if we broke those up such that there were less incoming and that the area
  // that it covered was sufficiently high that

  vector<tuple<int, int>> vertexes_sizes(storage.size());
  for (int i = 0; i < storage.size(); i++) {
    vertexes_sizes[i] = make_tuple(i, storage.get_vertex(i).size(&storage));
  }
  sort(vertexes_sizes.begin(), vertexes_sizes.end(),
       [](const auto &a, const auto &b) { return get<1>(a) > get<1>(b); });

  vector<int> covered(storage.size());

  // take the top sized vertexes.  These must be due to the incoming edge counts
  // as outgoing should all be the same number of edges.
}

template void super_nodes(const PMatrix<float> &matrix, dynamic_storage<float> &storage);
template void super_nodes(const PMatrix<double> &matrix, dynamic_storage<double> &storage);

template <typename float_t>
void shuffle_all_edges(const PMatrix<float_t> &matrix, dynamic_storage<float_t> &storage, int num_incoming_edges) {
  // sort the order in which edges are referred to so that we look at unique
  // areas as much as possible and can hopefully avoid expanding too much when
  // looking around nodes which are not near our desired answer.  Once we are
  // sufficiently close that we are going expand a node enterily, then this
  // should not have any impact on the runtime as we are not increasing the
  // number of neighbors and we are going to have to look at everything anyways.
  //
  // this can do something like take the first 10 vertices first.  That way it
  // still has stuff that is close getting looked at first?

  // this should take place on all edges, though in the dynamic storage save file
  // that does not include the all edges as an explicit saved object.
  // so that would need to be rewritten to duplicate that information

  // also select incoming edges which are unique so that we get the distance
  // search effect, though this should preven us from having to expand so many
  // neighbors.  So we are going to take the top k that are unique

  // assert(false); // don't use for now

  typedef typename dynamic_storage<float_t>::edge edge;
  typedef Eigen::Matrix<float_t, Eigen::Dynamic, 1> vecD;

  time_t start = time(NULL);
  uint processed_vertexes = 0;

  const uint batch_size = 1;

#ifdef CERTIFIEDCOSINE_USE_PARALLEL
#pragma omp parallel
#endif
  {
    vecD vector_sum(matrix.cols());
    vector<tuple<edge, float_t>> scores;

    const int omp_id = omp_get_thread_num();
    uint8_t printcnt = 0;

#ifdef CERTIFIEDCOSINE_USE_PARALLEL
#pragma omp for schedule(dynamic, 100)
#endif
    for (int i = 0; i < storage.size(); i++) {
      auto vx = storage.get_vertex(i);
      vector_sum.setZero();

      // insert all of the vertices into the neighbor list
      scores.clear();
      auto &outgoing = vx.outgoing_edges(&storage);
      for (auto e : outgoing) {
        scores.push_back(make_tuple(e, 0));
      }
      edge proof = outgoing[vx.proof_position(&storage)];
      outgoing.clear();

      int j;
      for (j = 0; j < batch_size; j++) {
        outgoing.push_back(get<0>(scores[j]));
        get<1>(scores[j]) = numeric_limits<float_t>::infinity();  // prevent this from getting picked again
        vector_sum += matrix.row(get<0>(scores[j]).id);           // this is the sum of these vertices
      }

      auto do_shuffle = [&](auto &scores, auto &save, int limit) {
        while (j < scores.size() && j < limit) {
          for (int k = 0; k < scores.size(); k++) {
            get<1>(scores[k]) += matrix.row(get<0>(scores[k]).id).dot(vector_sum);
          }
          vector_sum.setZero();
          // identify which ones are the best items to take, and then we are going to take a batch_size worth of items
          sort(scores.begin(), scores.end(), [](const auto &a, const auto &b) { return get<1>(a) < get<1>(b); });

          for (int k = 0; k < batch_size && j < scores.size() && j < limit; k++, j++) {
            save.push_back(get<0>(scores[k]));
            get<1>(scores[k]) = numeric_limits<float_t>::infinity();
            vector_sum += matrix.row(get<0>(scores[k]).id);
          }
        }
      };

      // shuffle the outgoing edges
      do_shuffle(scores, outgoing, scores.size());

      scores.clear();
      auto &incoming = vx.incoming_edges(&storage);
      for (auto e : incoming) {
        // filter out edges which are already in the outgoing set
        if (std::find_if(outgoing.begin(), outgoing.end(), [=](auto v) { return v.id == e.id; }) == outgoing.end())
          scores.push_back(make_tuple(e, 0));
      }
      for (auto e : outgoing) {
        vector_sum += matrix.row(e.id);
      }

      incoming.clear();
      // shuffle the incoming edges, they will come after the outgoing edges in the proof distance
      j = 0;
      do_shuffle(scores, incoming, num_incoming_edges);

      assert(proof.score == vx.proof_distance(&storage));

      processed_vertexes++;
      if (omp_id == 0 && printcnt++ % 50 == 0) {
        // print our a progress of how far this has gotten
        int secs = difftime(time(NULL), start);
        float done = ((float)processed_vertexes) / storage.size();
        printf("       \rProcessing edge sorting: %.2f%%, remaining mins: %i/%i", done * 100,
               (int)(secs / done * (1 - done) / 60), (int)(secs / done / 60));
        fflush(stdout);
      }
    }
  }
}

template void shuffle_all_edges(const PMatrix<float> &matrix, dynamic_storage<float> &storage, int num_incoming_edges);
template void shuffle_all_edges(const PMatrix<double> &matrix, dynamic_storage<double> &storage,
                                int num_incoming_edges);

template <typename float_t>
void make_smaller(const dynamic_storage<float_t> &input_storage, dynamic_storage<float_t> &output_storage,
                  uint new_num_neighbors) {
  time_t start = time(NULL);

  uint processed_vertexes = 0;

  output_storage.set_num_vertexes(input_storage.size());

  typedef typename dynamic_storage<float_t>::edge edge_p;
  struct edge : edge_p {
    edge(int d, float_t s) : edge_p(d, s) {}
    edge() : edge_p(-1, Consts<float_t>::worseScore) {}
    // sort the smallest values first so that we can remove the easily from the heap
    bool operator<(const edge &o) { return this->score > o.score; }
  };

#ifdef CERTIFIEDCOSINE_USE_PARALLEL
#pragma omp parallel
#endif
  {
    const int omp_id = omp_get_thread_num();
    uint8_t printcnt = 0;

#ifdef CERTIFIEDCOSINE_USE_PARALLEL
#pragma omp for schedule(dynamic, 50)
#endif
    for (uint i = 0; i < input_storage.size(); i++) {
      auto inp = input_storage.get_vertex(i);
      auto out = output_storage.get_vertex(i);

      // gaa hack
      auto iii = &inp.outgoing_edges(const_cast<dynamic_storage<float_t> *>(&input_storage));
      vector<edge> edges(*(vector<edge> *)iii);
      sort(edges.begin(), edges.end());

      auto &outgoing = out.outgoing_edges(&output_storage);
      outgoing.resize(min((size_t)new_num_neighbors, edges.size()));
      for (int j = 0; j < outgoing.size(); j++) {
        outgoing[j] = edges[j];
      }
      out.set_proof_distance(&output_storage, outgoing.size() - 1, outgoing[outgoing.size() - 1].score);
    }
  }
}

template void make_smaller(const dynamic_storage<float> &input_storage, dynamic_storage<float> &output_storage,
                           uint new_num_neighbors);
template void make_smaller(const dynamic_storage<double> &input_storage, dynamic_storage<double> &output_storage,
                           uint new_num_neighbors);

}  // namespace certified_cosine
