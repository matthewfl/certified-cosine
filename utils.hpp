#ifndef _FAST_VEC_UTILS_H
#define _FAST_VEC_UTILS_H

#include <assert.h>
#include <algorithm>
#include <array>
#include <tuple>
#include <vector>

#include <atomic>

namespace certified_cosine {

template <typename int_t, int size>
class LRUContains {
 private:
  std::array<int_t, size> backing;

 public:
  LRUContains() { backing.fill(-1); }

  bool operator()(int_t val) {
    if (backing[val % backing.size()] == val) {
      return true;
    } else {
      backing[val % backing.size()] = val;
      return false;
    }
  }

  bool contains(int_t val) { return backing[val % backing.size()] == val; }

  void insert(int_t val) {
    assert(val != -1);
    backing[val % backing.size()] = val;
  }

  void clear() { backing.fill(-1); }

  void remove(int_t val) {
    if (backing[val % backing.size()] == val) {
      backing[val % backing.size()] = -1;
    }
  }
};

template <typename int_t, int size>
class LRUCell {
 private:
  std::array<int_t, size> backing;

 public:
  LRUCell() { backing.fill(-1); }

  bool contains(int_t val) {
    for (int i = 0; i < size; i++) {
      if (backing[i] == val) {
        // move the key up given that it was just recently used
        for (; i > 0; i--) backing[i] = backing[i - 1];
        backing[0] = val;
        return true;
      }
    }
    return false;
  }

  void insert(int_t val) {
    int i = 0;
    for (; i < size - 1; i++)
      if (backing[i] == val) break;
    for (; i > 0; i--) backing[i] = backing[i - 1];
    backing[0] = val;
  }

  bool operator()(int_t val) {
    int i = 0;
    for (; i < size; i++)
      if (backing[i] == val) break;
    bool ret = i < size;
    for (; i > 0; i--) backing[i] = backing[i - 1];
    backing[0] = val;
    return ret;
  }

  void clear() { backing.fill(-1); }

  void remove(int_t val) {
    int i = 0;
    for (; i < size; i++)
      if (backing[i] == val) break;
    for (; i < size - 1; i++) backing[i] = backing[i + 1];
    backing[backing.size() - 1] = -1;
  }
};

template <typename int_t, int cell_size, int num_cells>
class LRUContainsMultiway {
 private:
  std::array<LRUCell<int_t, cell_size>, num_cells> backing;

 public:
  LRUContainsMultiway() {}
  bool contains(int_t val) { return backing[val % num_cells].contains(val); }
  void insert(int_t val) { return backing[val % num_cells].insert(val); }
  bool operator()(int_t val) { return backing[val % num_cells](val); }
  void clear() {
    for (auto& b : backing) b.clear();
  }
  void remove(int_t val) { backing[val % num_cells].remove(val); }
};

// https://stackoverflow.com/a/29195378/144600
// why is this not included somewhere in the standard library
class SpinLock {
  std::atomic_flag locked = ATOMIC_FLAG_INIT;

 public:
  void lock() {
    while (locked.test_and_set(std::memory_order_acquire)) {
      ;
    }
  }
  void unlock() { locked.clear(std::memory_order_release); }
};

template <int N>
class SpinLockN {
 private:
  SpinLock m_locks[N];

 public:
  inline void lock(int id) { m_locks[id % N].lock(); }
  inline void unlock(int id) { m_locks[id % N].unlock(); }
};

// value must implement:
//   `int v.get_key() const` method for the integer key, this must return -1 in the case that this
//   `bool is_empty() const` true in the case that this is a placeholder element
template <typename T>
class CuckooHashTable final {
 private:
  T* elements = nullptr;
  uint32_t size_ = 0;
  uint32_t offset_ = 0;  // by increasing offset, we are going to be unable to
                         // find elements that are currently inserted, so this
                         // functions as a soft reset

  inline uint32_t rehash(uint32_t x) {
    // based off the murmur3 hash avalanche to shuffle all of the bits
    x ^= x >> 16;
    x *= 0x85ebca6b;
    x ^= x >> 13;
    x *= 0xc2b2ae35;
    x ^= x >> 16;
    return x;
  }

  inline uint32_t mod_size(uint32_t x) {
    // return x % (size_ / 2);
    // the size will now be a power of 2, avoid the mod operator
    // as it requires using integer division where this should just be some bit masks
    return x & ((size_ >> 1) - 1);
  }

  void resize(const auto& maybe_drop_checker) {
    T* old = elements;
    std::vector<T> reinsert;
    const auto old_size = size_;
    const auto old_mask = (old_size / 2) - 1;
    size_ = size_ * 2;
    assert(__builtin_popcountll(size_) == 1);
    elements = new T[size_];

    const uint old_offset = offset_;
    offset_ = 0;
    for (uint32_t i = 0; i < old_size / 2; i++) {
      auto& v = old[i];
      if (v.get_key() != -1 && ((v.get_key() + old_offset) & old_mask) == i) {
        if (!insert_1(v, maybe_drop_checker)) {
          // then we are going to have to do something about this element to reshuffle this in elsewhere
          // (basically we want to avoid recursivly calling ourselves to resize)
          reinsert.push_back(v);
        }
      }
    }
    for (uint32_t i = old_size / 2; i < old_size; i++) {
      auto& v = old[i];
      if (v.get_key() != -1 && (((rehash(v.get_key()) + old_offset) & old_mask) + (old_size / 2)) == i) {
        if (!insert_1(v, maybe_drop_checker)) {
          reinsert.push_back(v);
        }
      }
    }

    delete[] old;

    // for elements that we were not able to directly reinsert into the hash table
    // this will perform reshuffling and possible further resizing as needed
    for (auto& v : reinsert) {
      insert_shuffle(v, maybe_drop_checker);  // this might call resize internally
    }
  }

  T* insert_1(const T& v, const auto& maybe_drop_checker) {
    // insert without a shuffle operation
    int k = v.get_key();
    uint i = mod_size(k + offset_);  // % (size_ / 2);
    if (elements[i].get_key() == -1 || elements[i].get_key() == k || mod_size(elements[i].get_key() + offset_) != i ||
        maybe_drop_checker(elements[i])) {
      elements[i] = v;
      return &elements[i];
    }
    i = mod_size(rehash(k) + offset_) + (size_ / 2);
    if (elements[i].get_key() == -1 || elements[i].get_key() == k ||
        (mod_size(rehash(elements[i].get_key()) + offset_) + (size_ / 2)) != i || maybe_drop_checker(elements[i])) {
      elements[i] = v;
      return &elements[i];
    }
    return nullptr;
  }

  void insert_shuffle(const T& v, const auto& maybe_drop_checker) {
    // then we are going to assume that there is already something that is in both of the buckets in this case
    // so we are just going to start trying to shuffle the elements until we are able to find something that
    // in the case that this fails
    T value = v;
    int k = v.get_key();
    const size_t size = size_ / 2;
    uint at = mod_size(k + offset_);
    uint64_t track = 0;  // approximate track which cells we have been in
    int cnt = size > 100 ? 10 : size / 100;
    while (true) {
      std::swap(elements[at], value);
      int k2 = value.get_key();
      if (k2 == -1 || k2 == k) return;  // then we inserted this element successfully
      if (at < size) {                  // check if this element should be keept in the table
        if (mod_size(k2 + offset_) != at) return;
      } else {
        if (mod_size(rehash(k2) + offset_) + size != at) return;
      }
      if (maybe_drop_checker(value)) return;

      k = k2;
      uint at2 = mod_size(k + offset_);
      if (at == at2) {
        at = mod_size(rehash(k) + offset_) + size;
      } else {
        // count down in the case that we have already been in this cell (approximatly)
        if (track & (1 << (k % 64)))
          if (--cnt <= 0) break;
        track |= (1 << (k % 64));

        at = at2;
      }
    }

    // in this case, we have done a lot of isnerts, and have still not found
    // some space where we are able to stash this element so we are goign to
    // increase the size of the table and try the insert procedure again
    resize(maybe_drop_checker);
    insert(value, maybe_drop_checker);
  }

 public:
  inline T* lookup(int k) {
    uint i = mod_size(k + offset_);
    uint j = mod_size(rehash(k) + offset_) + (size_ / 2);
    T* ret = nullptr;
    int a = elements[i].get_key(), b = elements[j].get_key();
    if (a == k) ret = &elements[i];
    if (b == k) ret = &elements[j];
    return ret;
  }

  T* insert(const T& v) {
    return insert(v, [](T& a) { return false; });
  }

  T* insert(const T& v, const auto& maybe_drop_checker) {
    assert(v.get_key() >= 0);
    T* ref;
    if (!(ref = insert_1(v, maybe_drop_checker))) {
      insert_shuffle(v, maybe_drop_checker);
      ref = lookup(v.get_key());
    }
    return ref;
  }

  void remove(int k) {
    uint i = mod_size(k + offset_);
    if (elements[i].get_key() == k) {
      elements[i] = T();
      return;
    }
    i = mod_size(rehash(k) + offset_) + (size_ / 2);
    if (elements[i].get_key() == k) {
      elements[i] = T();
    }
  }

  void prefetch(int k) {
    // nop
  }

  void soft_clear() {
    if (++offset_ == size_ / 2) {
      clear();
    }
  }

  void clear() {
    offset_ = 0;
    for (size_t i = 0; i < size_; i++) {
      elements[i] = T();
    }
  }

  CuckooHashTable() {
    elements = new T[32];
    size_ = 32;
  }

  ~CuckooHashTable() { delete[] elements; }
};

template <typename T>
class FlatTable final {
  T* elements = nullptr;
  uint32_t size_;
  uint32_t offset;

  inline uint32_t mod_size(int32_t x) {
    x -= offset;
    if (x < 0) x += size_;
    return x;
  }

 public:
  T* lookup(int k) {
    uint32_t l = mod_size(k);
    if (elements[l].get_key() == k) return &elements[l];
    return nullptr;
  }

  T* insert(const T& v) {
    uint32_t l = mod_size(v.get_key());
    elements[l] = v;
    return &elements[l];
  }

  T* insert(const T& v, const auto& maybe_drop_checker) {
    uint32_t l = mod_size(v.get_key());
    elements[l] = v;
    return &elements[l];
  }

  void remove(int k) {
    uint32_t l = mod_size(k);
    elements[k] = T();
  }

  void prefetch(int k) { __builtin_prefetch(&elements[mod_size(k)], 1, 1); }

  void soft_clear() {
    if (size_ == ++offset) clear();
  }

  void clear() {
    for (size_t i = 0; i < size_; i++) {
      elements[i] = T();
    }
    offset = 0;
  }

  FlatTable(uint32_t size) {
    offset = 0;
    size_ = size;
    elements = new T[size_];
    clear();
  }

  ~FlatTable() { delete[] elements; }
};

// T must implement
//   `compariable (float) get_value () const`
//   this will be able to identify the min and max element
//   T() must return an empty element
// based on details from: http://www.mhhe.com/engcs/compsci/sahni/enrich/c9/interval.pdf
template <typename T>
class IntervalHeap {
 private:
  // <min, max> maintained as a list of elements
  std::vector<std::tuple<T, T>> elements;
  size_t size_ = 0;

 public:
  void clear() {
    size_ = 0;
    elements.clear();
  }

#ifndef NDEBUG
  void check() {
    for (int i = 0; i < size_; i++) {
      if (i % 2 == 0)
        std::get<0>(elements[i / 2]).get_value();
      else
        std::get<1>(elements[i / 2]).get_value();
    }
  }
#endif

  inline size_t size() const { return size_; }

  void remove_max() {
    using namespace std;
    assert(size_ > 0);
    // take the last element and bubble it down
    if (size_ == 1) {
      elements.clear();
      size_ = 0;
    } else if (size_ == 2) {
      get<1>(elements[0]) = T();
      size_--;
    } else {
      T v;
      if (size_ % 2 == 1) {
        v = get<0>(elements.back());
        elements.pop_back();  // reduce the size of the heap
      } else {
        v = get<1>(elements.back());
        get<1>(elements.back()) = T();
      }
      auto vval = v.get_value();
      int i = 1,   // current node
          ci = 2;  // child of i
      while (ci <= elements.size()) {
        if (ci < elements.size()) {
          if (ci == elements.size() - 1 && size_ % 2 == 0) {
            if (get<1>(elements[ci - 1]).get_value() < get<0>(elements[ci]).get_value()) ci++;
          } else {
            if (get<1>(elements[ci - 1]).get_value() < get<1>(elements[ci]).get_value()) ci++;
          }
        }

        assert(ci - 1 < elements.size());
        if (ci == elements.size() && size_ % 2 == 0) {
          if (vval >= get<0>(elements[ci - 1]).get_value()) break;
        } else {
          if (vval >= get<1>(elements[ci - 1]).get_value()) break;
        }

        // move the element up
        if (ci == elements.size() && size_ % 2 == 0) {
          get<1>(elements[i - 1]) = get<0>(elements[ci - 1]);
        } else {
          get<1>(elements[i - 1]) = get<1>(elements[ci - 1]);

          if (vval < get<0>(elements[ci - 1]).get_value()) {
            std::swap(v, get<0>(elements[ci - 1]));
            vval = v.get_value();
          }
        }

        // move down a level
        i = ci;
        ci *= 2;
      }
      if (i == elements.size() && size_ % 2 == 0) {
        get<0>(elements[i - 1]) = v;
      } else {
        get<1>(elements[i - 1]) = v;
      }
      size_--;
    }
  }

  void remove_min() {
    using namespace std;
    assert(size_ > 0);
    if (size_ == 1) {
      elements.clear();
      size_ = 0;
    } else if (size_ == 2) {
      get<0>(elements[0]) = get<1>(elements[0]);
      get<1>(elements[0]) = T();
      size_--;
    } else {
      T v;
      if (size_ % 2 == 1) {
        v = get<0>(elements.back());
        elements.pop_back();  // reduce the size of the heap
      } else {
        v = get<1>(elements.back());
        get<1>(elements.back()) = T();
      }
      auto vval = v.get_value();
      uint i = 1, ci = 2;
      while (ci <= elements.size()) {
        if (ci < elements.size() && get<0>(elements[ci - 1]).get_value() > get<0>(elements[ci]).get_value()) ci++;

        assert(ci - 1 < elements.size());
        if (vval <= get<0>(elements[ci - 1]).get_value()) break;

        // move the element up
        get<0>(elements[i - 1]) = get<0>(elements[ci - 1]);
        if (!(ci == elements.size() && size_ % 2 == 0) && vval > get<1>(elements[ci - 1]).get_value()) {
          std::swap(v, get<1>(elements[ci - 1]));
          vval = v.get_value();
        }
        // move down a level
        i = ci;
        ci *= 2;
      }
      get<0>(elements[i - 1]) = v;
      size_--;
    }
  }

  void insert(const T& value) {
    using namespace std;
    if (size_ == 0) {
      elements.push_back(make_tuple(value, T()));
      size_++;
    } else if (size_ == 1) {
      get<1>(elements[0]) = value;
      if (get<1>(elements[0]).get_value() < get<0>(elements[0]).get_value()) {
        std::swap(get<1>(elements[0]), get<0>(elements[0]));
      }
      size_++;
    } else {
      bool minHeap;
      auto val = value.get_value();
      if (size_ % 2 == 1) {
        // odd number of elements
        if (val < get<0>(elements.back()).get_value()) {
          // make space for this item in the min value side
          std::swap(get<1>(elements.back()), get<0>(elements.back()));
          minHeap = true;
        } else {
          minHeap = false;
        }
      } else {
        // even number of elements
        elements.push_back(make_tuple(T(), T()));
        if (val < get<0>(elements[elements.size() / 2 - 1]).get_value()) {
          minHeap = true;
        } else {
          minHeap = false;
        }
      }
      assert(val != -1);

      if (minHeap) {
        int i = elements.size();
        while (i != 1 && val < get<0>(elements[i / 2 - 1]).get_value()) {
          get<0>(elements[i - 1]) = get<0>(elements[i / 2 - 1]);
          i /= 2;
        }
        get<0>(elements[i - 1]) = value;
      } else {
        int i = elements.size();
        while (i != 1 && val > get<1>(elements[i / 2 - 1]).get_value()) {
          get<1>(elements[i - 1]) = get<1>(elements[i / 2 - 1]);
          i /= 2;
        }
        get<1>(elements[i - 1]) = value;
        if (size_ % 2 == 0) {
          std::swap(get<1>(elements.back()), get<0>(elements.back()));
        }
      }
      size_++;
    }
  }

  inline bool is_empty() const { return elements.size() == 0; }

  inline const T& min() const {
    using namespace std;
    assert(elements.size() > 0);
    return get<0>(elements[0]);
  }
  inline const T& max() const {
    // TODO: might want to change this given that we are having to branch here,
    // and we are currently using the max as the entry that we are interested in
    // seeing first
    using namespace std;
    assert(elements.size() > 0);
    if (size_ == 1) return get<0>(elements[0]);
    return get<1>(elements[0]);
  }

  auto begin() { return iterator(elements, 0); }

  auto end() { return iterator(elements, size_); }

  class iterator {
   private:
    typedef typename std::vector<std::tuple<T, T>>::iterator viter;
    const std::vector<std::tuple<T, T>>& ref;
    size_t at;

    inline void inc() { at++; }
    iterator(const std::vector<std::tuple<T, T>>& ref, size_t at) : ref(ref), at(at) {}
    friend class IntervalHeap;

   public:
    typedef iterator self_type;
    typedef const T value_type;
    typedef const T& reference;
    typedef T* const pointer;

    self_type operator++() {
      self_type i = *this;
      inc();
      return i;
    }
    self_type& operator++(int) {
      inc();
      return *this;
    };
    reference operator*() {
      if (at % 2 == 0)
        return std::get<0>(ref[at / 2]);
      else
        return std::get<1>(ref[at / 2]);
    }
    pointer operator->() { return at % 2 == 0 ? &std::get<0>(ref[at / 2]) : &std::get<1>(ref[at / 2]); }
    bool operator==(self_type& o) { return o.at == at; }
    bool operator!=(self_type& o) { return !(*this == o); }
  };
};

}  // namespace certified_cosine

// extern "C" void openblas_set_num_threads(int);

#endif
