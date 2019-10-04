#include "catch.hpp"

#include <algorithm>
using namespace std;

#define private public
#include "storage.hpp"
#undef private

#include "pre_processing.hpp"

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/normal_distribution.hpp>
static boost::random::normal_distribution<float> nd;
static boost::random::mt19937 rng(0);

using namespace certified_cosine;

static auto random_matrix(int rows, int cols) {
  return Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>::Zero(rows, cols)
      .unaryExpr([&](auto f) -> float { return nd(rng); })
      .eval();
}

TEST_CASE("dynamic storage", "[storage]") {
  dynamic_storage<float> storage;
  typedef decltype(storage)::edge edge;
  storage.set_num_vertexes(10);

  SECTION("basic") {
    decltype(storage)::vertex vx = storage.get_vertex(0);
    REQUIRE(vx.size(&storage) == 0);

    vx.outgoing_edges(&storage).push_back(edge(5, .2));
    REQUIRE(vx.outgoing_edges(&storage).size() == 1);

    vx.incoming_edges(&storage).push_back(edge(6, .1));
    vx.incoming_edges(&storage).push_back(edge(5, .2));
    REQUIRE(vx.incoming_edges(&storage).size() == 2);

    vx.build_all_edges(&storage);

    auto opaque = vx.neighbor_opaque(&storage);
    REQUIRE(vx.size(&storage) == 2);
    REQUIRE(vx.neighbor(&storage, opaque, 0) == 5);
    REQUIRE(vx.neighbor(&storage, opaque, 1) == 6);
  }
}

TEST_CASE("compact storage", "[storage]") {
  dynamic_storage<float> storage;
  typedef decltype(storage)::edge edge;

  // first build a dynamic storage
  storage.set_num_vertexes(10);
  for (int i = 0; i < storage.size(); i++) {
    auto vx = storage.get_vertex(i);
    auto &ic = vx.incoming_edges(&storage);
    auto &oc = vx.outgoing_edges(&storage);

    // normally these edges would be sorted
    for (int j = 0; j < 5; j++) {
      ic.push_back(edge(j, .5 - j * .1));
      oc.push_back(edge(j + 5, 1 - j * .1));
    }

    vx.set_proof_distance(&storage, 2, oc[2].score);

    vx.build_all_edges(&storage);
  }

  compact_storage<float> compact;
  storage.BuildCompactStorage(compact);

  REQUIRE(compact.size() == storage.size());

  for (int i = 0; i < storage.size(); i++) {
    auto vx = compact.get_vertex(i);
    auto vxd = storage.get_vertex(i);
    REQUIRE(vx.proof_distance(&compact) == vxd.proof_distance(&storage));

    auto opaque = vx.neighbor_opaque(&compact);
    REQUIRE(vx.neighbor(&compact, opaque, 0) == 5);
    REQUIRE(vx.neighbor(&compact, opaque, 2) == 7);
    REQUIRE(vx.neighbor(&compact, opaque, 6) == 1);
    REQUIRE(vx.size(&compact) > 0);
  }
}

namespace std {
bool operator==(const dynamic_storage<float>::edge_s a, const dynamic_storage<float>::edge_s b) {
  return a.id == b.id && a.score == b.score;
}
}  // namespace std

TEST_CASE("storage save load", "[storage]") {
  auto testMat = random_matrix(1123, 15);
  auto norms = testMat.rowwise().norm().eval();
  testMat.array().colwise() /= norms.array();

  // getting random stuff into the storage etc
  dynamic_storage<float> storage;
  preprocess<float>(testMat, storage);

  storage.Save("/tmp/fast_vectors_test_storage");

  dynamic_storage<float> storage2;
  storage2.Load("/tmp/fast_vectors_test_storage");

  // now check that these are the same
  REQUIRE(storage.size() == storage2.size());

  auto sarr1 = storage.starting_arr();
  auto sarr2 = storage2.starting_arr();
  REQUIRE(equal(sarr1.begin(), sarr1.end(), sarr2.begin()));

  for (int i = 0; i < storage.size(); i++) {
    auto vx1 = storage.get_vertex(i);
    auto vx2 = storage2.get_vertex(i);

    REQUIRE(vx1.size(&storage) == vx2.size(&storage2));
    REQUIRE(vx1.proof_distance(&storage) == vx2.proof_distance(&storage2));

    const auto &vn1 = vx1.get_all_edges(&storage);
    const auto &vn2 = vx2.get_all_edges(&storage2);
    REQUIRE(equal(vn1.begin(), vn1.end(), vn2.begin()));
  }

  compact_storage<float> compact2;
  storage.BuildCompactStorage(compact2);
  compact2.Save("/tmp/fast_vectors_test_storage_compact");

  compact_storage<float> compact;
  compact.Load("/tmp/fast_vectors_test_storage_compact");

  REQUIRE(equal(sarr1.begin(), sarr1.end(), compact.starting.begin()));

  REQUIRE(compact.size() == storage.size());
  REQUIRE(compact.proof_distance_size() == storage.proof_distance_size());

  for (int i = 0; i < storage.size(); i++) {
    auto vx1 = storage.get_vertex(i);
    auto vx2 = compact.get_vertex(i);

    auto o1 = vx1.neighbor_opaque(&storage);
    auto o2 = vx2.neighbor_opaque(&compact);

    REQUIRE(vx1.size(&storage) == vx2.size(&compact));
    REQUIRE(vx1.proof_distance(&storage) == vx2.proof_distance(&compact));
    for (int j = 0; j < vx2.size(&compact); j++) {
      REQUIRE(vx1.neighbor(&storage, o1, j) == vx2.neighbor(&compact, o2, j));
    }
  }
}
