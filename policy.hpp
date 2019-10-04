#ifndef _CERTIFIEDCOSINE_POLICY
#define _CERTIFIEDCOSINE_POLICY

/**
 * Policies control when the lookup procedure is done.  They are "notified"
 * anytime something new is expanded or anytime a new proof has been
 * constructed.
 *
 * Policies implement a few basic methods which constrol how the lookup is performed
 *  typename float_t
 *         The float type that this policy supports (either float or double)
 *
 *  bool expand(int id, float_t score)
 *         Everytime that a new vector is expanded, this will be called with the vectors id (index in the embedding
 * matrix) and the score representing the cosine similarity with the query. Returning true will STOP the current search.
 * Presummaly this should correspond with exceeding a budget. Note: score is in terms of cosine similarity.  A value of
 * 1 corresponds with a "distance" of 0, and -1 is the furthest possible point.  (distance = 1 - score)
 *
 *  float_t proof_distance() const
 *         The size of $\mathcal{N}_{\hat{v}}$ which controls how large the neighborhood around the query $q$ is that we
 * are trying to construct a proof for.  In the case of certifiying the top-1 nearest neighbor, then this should just
 * return the $q^T \hat{v}$.  For certifying the top-k nearest neighbors, then this should return $q^T v^{(k)}$. Note:
 * this the distances is returned in terms of cosine similiary.  So 1 corresponds with the _smallest_ possible
 * $\mathcal{N}_{\hat{v}}$ and -1 is the largest possible area.
 */

#include <assert.h>

#include "constants.hpp"
#include "utils.hpp"

#include <unordered_set>

namespace certified_cosine {

template <typename float_T>
struct OneBestPolicy {
  typedef float_T float_t;

  int id = -1;
  float_t distance = Consts<float_t>::infWorseScore;
  inline bool expand(int id, float_t score) {
    if (compareScore(score, distance)) {
      distance = score;
      this->id = id;
    }

    return false;
  }

  inline float_t proof_distance() const {
    // the distance that this proving system is looking for atm
    return distance;
  }

  bool got_proof() const {
    // this just keeps looking until it found the best item,
    // so when it terminates, that means that it must have constructed the proof
    return true;
  }
};

template <typename float_T>
struct NBestPolicy {
  typedef float_T float_t;

  struct located {
    float_t score;
    int id;
    located() : id(-1) {}
    located(int id, float_t score) : score(score), id(id) {}
    float_t get_value() const { return score; }
  };

  int n;
  IntervalHeap<located> items;

  inline bool add_to_items(int id, float_t score) {
    for (auto &a : items) {
      if (a.id == id) return false;  // then already added
    }
    assert(false);
    items.insert(located(id, score));
    return true;
  }

  inline bool expand(int id, float_t score) {
    if (items.size() < n) {
      add_to_items(id, score);
      return false;
    } else {
      if (compareScore(score, items.min().score)) {
        if (add_to_items(id, score)) items.remove_min();
      }
      return false;
    }
  }

  inline float_t proof_distance() const {
    assert(items.size() > 0);
    if (items.size() < n) return Consts<float_t>::infWorseScore;
    return items.min().score;
  }

  bool got_proof() const { return true; }

  NBestPolicy(int n) : n(n) {}
};

template <typename float_T>
struct NBestSingleProof : NBestPolicy<float_T> {
  typedef float_T float_t;

  inline float_t proof_distance() const {
    assert(this->items.size() > 0);
    return this->items.max().score;
  }

  NBestSingleProof(int n) : NBestPolicy<float_t>(n) {}
};

template <typename parent>
struct CountExpandPolicy : parent {
  typedef typename parent::float_t float_t;

  int count = 0;

  inline bool expand(int id, float_t score) {
    count++;
    return parent::expand(id, score);
  }
};

template <typename float_T>
struct CountingTillBest {
  typedef float_T float_t;

  int count = 0;
  int count_located = -1;

  int id = -1;
  float_t distance = Consts<float_t>::infWorseScore;
  inline bool expand(int id, float_t score) {
    count++;
    if (compareScore(score, distance)) {
      distance = score;
      this->id = id;
      count_located = count;
    }

    return false;
  }

  inline float_t proof_distance() const { return distance; }

  bool got_proof() const { return true; }
};

template <typename float_T>
struct CountingNBestPolicy {
  typedef float_T float_t;

  struct located {
    float_t score;
    int id;
    located() : id(-1) {}
    located(int id, float_t score) : score(score), id(id) {}
    float_t get_value() const { return score; }
  };

  int n;
  IntervalHeap<located> items;

  bool add_to_items(int id, float_t score) {
    for (auto &a : items) {
      if (a.id == id) return false;  // then already added
    }
    items.insert(located(id, score));
    return true;
  }

  int count = 0;
  int count_located = -1;

  inline bool expand(int id, float_t score) {
    count++;
    if (items.size() < n) {
      if (add_to_items(id, score)) count_located = count;
      return false;
    } else {
      if (compareScore(score, items.min().score)) {
        if (add_to_items(id, score)) {
          items.remove_min();
          count_located = count;
        }
      }
      return false;
    }
  }

  inline float_t proof_distance() const {
    assert(items.size() > 0);
    if (items.size() < n) return Consts<float_t>::infWorseScore;
    return items.min().score;
  }

  bool got_proof() const { return true; }

  CountingNBestPolicy(int n) : n(n) {}
};

template <typename parent>
struct LimitExpand : parent {
  typedef typename parent::float_t float_t;
  int limit;

  inline bool expand(int id, float_t score) { return parent::expand(id, score) || this->count > limit; }

  bool got_proof() const { return this->count <= limit; }

  LimitExpand(int a) : limit(a) {}
  LimitExpand(int a, int b) : limit(a), parent(b) {}
};

template <typename parent>
struct ApproximatePolicy : parent {
  // make the proof distance have a shrinking radius based off how long we have
  // run, so that we don't spend too long.  In this case we are not constructing
  // the "true" proof, but rather for a smaller area

  typedef typename parent::float_t float_t;

  inline float_t proof_distance() const {
    float_t p = parent::proof_distance();
    return ((float_t)1.0) - (((float_t)1.0) - p) * (((float_t)1.0) - ((float_t)this->count) / (this->limit + 500));
  }

  ApproximatePolicy(int a) : parent(a) {}
  ApproximatePolicy(int a, int b) : parent(a, b) {}
};

template <typename parent>
struct ProveBest : parent {
  typedef typename parent::float_t float_t;

  inline float_t proof_distance() const {
    assert(this->items.size() > 0);
    // return the max score instead of the min score, as we are only going to
    // prove the 1 best in this case instead of all of the points
    return this->items.max().score;
  }

  ProveBest() : parent() {}
  ProveBest(int a) : parent(a) {}
  ProveBest(int a, int b) : parent(a, b) {}
};

template <typename float_T>
struct WrappedPolicy {
  typedef float_T float_t;

 private:
  struct WrappedBased {
    virtual bool expand(int id, float_t score) = 0;
    virtual float_t proof_distance() const = 0;
    virtual bool got_proof() const = 0;
  };

  template <typename T>
  struct WrappedT : WrappedBased {
    T *t;
    WrappedT(T *t) : t(t) {}
    bool expand(int id, float_t score) override { return t->expand(id, score); }
    float_t proof_distance() override { return t->proof_distance(); }
    bool got_proof() override { return t->got_proof(); }
  };

  uint8_t wrapped[sizeof(WrappedT<WrappedBased>)];

 public:
  template <typename T>
  WrappedPolicy(T *t) {
    new (wrapped) WrappedT<T>(t);
  }

  bool expand(int id, float_t score) { return ((WrappedBased *)wrapped)->expand(id, score); }

  float_t proof_distance() const { return ((WrappedBased *)wrapped)->proof_distance(); }

  bool got_proof() const { return ((WrappedBased *)wrapped)->got_proof(); }
};

template <typename parent>
struct DebugPolicy : parent {
  typedef typename parent::float_t float_t;

  std::unordered_set<int> opened;
  bool expand(int id, float_t score) {
    // ensure that we do not repeat looking at the same vertex
    assert(!opened.count(id));
    opened.emplace(id);
    return parent::expand(id, score);
  }
};

}  // namespace certified_cosine

#endif
