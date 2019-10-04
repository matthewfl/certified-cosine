#include "catch.hpp"

#include "utils.hpp"

#include <unordered_set>

// test data structure for the cuckoo hashing
struct vv {
  int k;
  int v = 0;
  int get_key() const { return k; }  // this needs to return -1 in the case that there is nothing here
  int get_value() const {
    assert(k != -1);
    return v;
  }
  bool is_empty() const { return k == -1; }
  vv() : k(-1) {}
  vv(int i) : k(i) { assert(i >= 0 && i < 200000); }
  vv(int i, int vv) : k(i), v(vv) { assert(i >= 0 && i < 200000); }
};

using namespace certified_cosine;

TEST_CASE("Cuckoo Hashing", "[cuckoo][hashing][datastructure]") {
  CuckooHashTable<vv> table;

  for (int i = 0; i < 6; i++) table.insert(vv(i));

  SECTION("lookup") {
    REQUIRE(table.lookup(5) != nullptr);
    REQUIRE(table.lookup(6) == nullptr);
  }

  SECTION("remove") {
    table.remove(5);
    REQUIRE(table.lookup(5) == nullptr);
  }

  SECTION("clear") {
    REQUIRE(table.lookup(5) != nullptr);
    table.clear();
    REQUIRE(table.lookup(5) == nullptr);
  }

  SECTION("soft clear") {
    REQUIRE(table.lookup(5) != nullptr);
    table.soft_clear();
    REQUIRE(table.lookup(5) == nullptr);

    table.insert(vv(4));
    REQUIRE(table.lookup(4) != nullptr);
    REQUIRE(table.lookup(5) == nullptr);

    for (int i = 0; i < 20; i++) {
      table.soft_clear();
      REQUIRE(table.lookup(4) == nullptr);
    }
  }

  SECTION("override") {
    REQUIRE(table.lookup(5)->get_value() == 0);
    table.insert(vv(5, 5));
    REQUIRE(table.lookup(5)->get_value() == 5);
  }

  SECTION("lots of items") {
    table.soft_clear();
    for (int i = 0; i < 200; i++) {
      table.insert(vv(i * 1000));
    }

    for (int i = 0; i < 200; i++) {
      REQUIRE(table.lookup(i * 1000) != nullptr);
      REQUIRE(table.lookup(i * 1000 + 1) == nullptr);
    }

    table.soft_clear();
    for (int i = 0; i < 200; i++) {
      REQUIRE(table.lookup(i * 1000) == nullptr);
    }
  }
}

TEST_CASE("Interval Heap", "[heap][datastructure]") {
  IntervalHeap<vv> heap;

  heap.insert(vv(2, 2));
  heap.insert(vv(3, 3));

  SECTION("basic") {
    REQUIRE(heap.max().get_key() == 3);
    REQUIRE(heap.min().get_key() == 2);
  }

  SECTION("clear") {
    REQUIRE(!heap.is_empty());
    heap.clear();
    REQUIRE(heap.is_empty());
  }

  SECTION("inserts basic") {
    heap.clear();
    for (int i = 0; i < 100; i++) {
      heap.insert(vv(i, i * 2));
    }
    REQUIRE(heap.max().get_key() == 99);
    REQUIRE(heap.min().get_key() == 0);
  }

  SECTION("remove") {
    heap.clear();
    for (int i = 0; i < 100; i++) {
      heap.insert(vv(i, i * 3));
    }

    SECTION("min") {
      for (int i = 0; i < 100; i++) {
        REQUIRE(heap.max().get_key() == 99);
        REQUIRE(heap.min().get_key() == i);
        heap.remove_min();
      }
    }

    SECTION("max") {
      for (int i = 99; i >= 0; i--) {
        REQUIRE(heap.min().get_key() == 0);
        REQUIRE(heap.max().get_key() == i);
        heap.remove_max();
      }
    }
  }

  SECTION("iterator") {
    using namespace std;
    heap.clear();
    for (int i = 0; i < 100; i++) {
      heap.insert(vv(i, i));
    }
    for (int i = 75; i < 150; i++) {
      bool d = false;
      for (auto a : heap)
        if (a.k == i) d = true;
      if (!d) heap.insert(vv(i, i - 75));
    }
    unordered_set<int> output;
    for (auto &a : heap) {
      assert(!output.count(a.k));
      output.insert(a.k);
    }

    assert(output.size() == 150);

    for (int i = 0; i < 150; i++) {
      REQUIRE(output.count(i));
    }
  }
}
