#ifndef _CERTIFIEDCOSINE_STORAGE
#define _CERTIFIEDCOSINE_STORAGE

/**
 * The storage for the neighbor list data structure.
 *
 *  The dynamic_storage class is intended for preprocessing, it does not throw
 *  away any possibly useful parameters and thus takes more space.  Also, the
 *  dynamic storage class internally uses vectors of vectors which allows it to
 *  be resized at the expense of more pointer accesses.  It can be directly used
 *  with the lookup operation as it shares the same interface as the compact storage.
 *
 * compact_storage is intended to use a little space as possible.  Anything that
 * is not directly used by the lookup procedure has been removed.  Edges are
 * stored as a single large and offsets are computed as needed.
 */

#include <assert.h>
#include <algorithm>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include "constants.hpp"

namespace certified_cosine {

using namespace std;

template <typename float_t>
class compact_storage;

template <typename float_t>
class dynamic_storage;

template <typename float_t>
class compact_storage {
 private:
  static constexpr uint num_mapped = 512;

  struct vertex_storage {
    uint16_t neighbor_index;
    uint16_t num_neighbors;
    float_t proof_distance;
  };

  vector<uint> neighbors;

  vector<uint> vertexes_offsets;
  vector<vertex_storage> vertexes;

  friend class dynamic_storage<float_t>;

  vector<int> starting;

  int proof_distance_set_size = -1;

 public:
  class vertex {
    int id;
#ifndef NDEBUG

    const compact_storage *d_self;
#endif
    friend struct compact_storage;
    vertex(int id, const compact_storage *self)
        : id(id)
#ifndef NDEBUG
          ,
          d_self(self)
#endif
    {
    }

   public:
    vertex()
        : id(-1)
#ifndef NDEBUG
          ,
          d_self((compact_storage *)-1)
#endif
    {
    }
    // these methods take a reference to the storage object rather than
    // encapsuating that in the vertex class.  The reason is that we are going
    // to be storing many instances of this class whenever we are performing
    // search and duplicating this pointer would be wasteful
    int size(const compact_storage *self) const {
      assert(d_self == self);
      return self->vertexes[id].num_neighbors;
    }
    float_t proof_distance(const compact_storage *self) const {
      assert(d_self == self);
      return self->vertexes[id].proof_distance;
    }
    const uint *neighbor_opaque(const compact_storage *self) {
      assert(d_self == self);
      return &self->neighbors[self->vertexes_offsets[id / self->num_mapped] + self->vertexes[id].neighbor_index];
    }
    int neighbor(const compact_storage *self, const uint *opaque, int a) const {
      assert(d_self == self);
      return opaque[a];
    }
    int get_id() const { return id; }
  };

  vertex get_vertex(int id) const {
    assert(id < vertexes.size());
    return vertex(id, this);
  }

  int get_starting(int signature) const { return starting[signature & (starting.size() - 1)]; }

  int proof_distance_size() const { return proof_distance_set_size; }

  size_t size() const { return vertexes.size(); }

  void Save(ostream &file) {
    file << "certified_cosine_compact" << endl;
    file << vertexes.size() << " " << neighbors.size() << " " << vertexes_offsets.size() << " " << starting.size()
         << " " << proof_distance_set_size << "\n";
    file.write((const char *)vertexes.data(), sizeof(typename decltype(vertexes)::value_type) * vertexes.size());
    file.write((const char *)vertexes_offsets.data(),
               sizeof(typename decltype(vertexes_offsets)::value_type) * vertexes_offsets.size());
    file.write((const char *)neighbors.data(), sizeof(typename decltype(neighbors)::value_type) * neighbors.size());
    file.write((const char *)starting.data(), sizeof(typename decltype(starting)::value_type) * starting.size());
  }

  void Load(istream &file) {
    string self_name;
    file >> self_name;
    // this needs to be able to return error codes or something???
    assert(self_name == "certified_cosine_compact");
    uint nvertexes, nneighbors, noffsets, nstarting;
    file >> nvertexes >> nneighbors >> noffsets >> nstarting >> proof_distance_set_size;
    vertexes.resize(nvertexes);
    neighbors.resize(nneighbors);
    vertexes_offsets.resize(noffsets);
    starting.resize(nstarting);
    char zz = file.get();  // ignore new line
    assert(zz == '\n');
    (void)zz;
    file.read((char *)vertexes.data(), sizeof(typename decltype(vertexes)::value_type) * vertexes.size());
    file.read((char *)vertexes_offsets.data(),
              sizeof(typename decltype(vertexes_offsets)::value_type) * vertexes_offsets.size());
    file.read((char *)neighbors.data(), sizeof(typename decltype(neighbors)::value_type) * neighbors.size());
    file.read((char *)starting.data(), sizeof(typename decltype(starting)::value_type) * starting.size());
  }

  void Save(string fname) {
    // this should just seralize the edge vectors
    ofstream file(fname);
    Save(file);
  }

  void Load(string fname) {
    ifstream file(fname);
    Load(file);
  }
};

template <typename float_t>
class dynamic_storage {
 public:
  struct edge_s {
    int id;
    float_t score;
    edge_s() : id(-1), score(Consts<float_t>::worseScore) {}
    edge_s(int d, float_t s) : id(d), score(s) { assert(score >= Consts<float_t>::worseScore); }
    edge_s(const edge_s &o) : id(o.id), score(o.score) {}
  };

  typedef edge_s edge;

 private:
  struct vertex_s {
    vector<edge_s> all_edges;
    vector<edge_s> outgoing_edges;
    vector<edge_s> incoming_edges;
    // this should just be the distance of the outgoing_edges?
    // well I suppose that there might be some other things that are added along the way
    int proof_required = -1;
    float_t proof_distance = Consts<float_t>::invalid;
    vertex_s() {}
  };

  vector<vertex_s> vertexes;

  vector<int> starting = {0};  // unless set, just start at the zero'th entry

  int proof_distance_set_size = -1;

 public:
  class vertex {
   private:
    int id;
#ifndef NDEBUG
    const dynamic_storage *d_self;
#endif
    vertex(int id, const dynamic_storage *self)
        : id(id)
#ifndef NDEBUG
          ,
          d_self(self)
#endif
    {
    }

   public:
    vertex()
        : id(-1)
#ifndef NDEBUG
          ,
          d_self((dynamic_storage *)-1)
#endif
    {
    }

    friend class dynamic_storage<float_t>;

    // the minimal interface for the lookup only operation
    int size(const dynamic_storage *self) const {
      assert(d_self == self);
      return self->vertexes[id].all_edges.size();
    }
    float_t proof_distance(const dynamic_storage *self) const {
      assert(d_self == self);
      const auto &v = self->vertexes[id];
      return v.proof_distance;
    }

    int neighbor_opaque(const dynamic_storage *self) {
      // not used
      assert(d_self == self);
      return 0;
    }

    int neighbor(const dynamic_storage *self, int opaque, int a) const {
      assert(d_self == self);
      return self->vertexes[id].all_edges[a].id;
    }

    const edge_s &neighbor_dist(const dynamic_storage *self, int a) const {
      assert(d_self == self);
      return self->vertexes[id].all_edges[a];
    }

    int proof_position(const dynamic_storage *self) const {
      assert(d_self == self);
      const auto &v = self->vertexes[id];
      return v.proof_required;
    }

    int get_id() const { return id; }

    // operations required for inserting values
    void set_proof_distance(dynamic_storage *self, int id, float_t distance) {
      assert(d_self == self);
      auto &v = self->vertexes[this->id];
      assert(0 <= id && id < v.outgoing_edges.size());
      v.proof_required = id;
      v.proof_distance = distance;
      assert(v.outgoing_edges[id].score <= distance);  // check that this is consistent
    }

    vector<edge_s> &outgoing_edges(dynamic_storage *self) {
      assert(d_self == self);
      auto &v = self->vertexes[id];
      return v.outgoing_edges;
    }

    vector<edge_s> &incoming_edges(dynamic_storage *self) {
      assert(d_self == self);
      auto &v = self->vertexes[id];
      return v.incoming_edges;
    }

    const vector<edge_s> &get_all_edges(dynamic_storage *self) {
      assert(d_self == self);
      return self->vertexes[id].all_edges;
    }

    void build_all_edges(dynamic_storage *self) {
      assert(d_self == self);
      auto &v = self->vertexes[id];
      v.all_edges.clear();
      v.all_edges = v.outgoing_edges;  // copy
      // now we are going to have to copy over all of the edges which are not contained
      for (auto e : v.incoming_edges) {
        if (std::find_if(v.outgoing_edges.begin(), v.outgoing_edges.end(), [=](auto v) { return v.id == e.id; }) ==
            v.outgoing_edges.end())
          v.all_edges.push_back(e);
      }
      v.all_edges.shrink_to_fit();
    }
  };

  vertex get_vertex(int id) const {
    assert(id < vertexes.size());
    return vertex(id, this);
  }

  void set_num_vertexes(int num_vertexes, int proof_distance_size = -1) {
    vertexes.resize(num_vertexes);
    proof_distance_set_size = proof_distance_size;
  }

  size_t size() const { return vertexes.size(); }

  int proof_distance_size() const { return this->proof_distance_set_size; }

  int get_starting(int signature) const { return starting[signature & (starting.size() - 1)]; }

  vector<int> &starting_arr() { return starting; }

  void BuildCompactStorage(compact_storage<float_t> &compact) {
    // load the edges represented by this dynamic_storage into the compact
    // storage.

    uint64_t num_edges = 0;
    for (auto &v : vertexes) {
      num_edges += v.all_edges.size();
    }
    // there are only 32 bit ints for address the items so make sure that we did not overflow that
    // this should probably never happen
    assert(num_edges < std::numeric_limits<int32_t>::max());

    compact.neighbors.resize(num_edges);
    compact.vertexes.resize(vertexes.size());
    compact.vertexes_offsets.resize(vertexes.size() / compact.num_mapped + 1);
    compact.starting = starting;
    compact.proof_distance_set_size = proof_distance_set_size;
    uint e = 0;
    uint g = 0;
    uint o = 0;
    compact.vertexes_offsets[0] = 0;
    for (int i = 0; i < vertexes.size(); i++) {
      auto &v = vertexes[i];
      if (i / compact.num_mapped != o) {
        g = e;
        compact.vertexes_offsets[++o] = g;
      }
      compact.vertexes[i].proof_distance = v.proof_distance;
      assert(v.all_edges.size() < (1 << 15));
      compact.vertexes[i].num_neighbors = v.all_edges.size();
      compact.vertexes[i].neighbor_index = e - g;
      for (edge_s j : v.all_edges) {
        compact.neighbors[e++] = j.id;
      }
    }
  }

  void Save(string fname) {
    ofstream file(fname, ios::binary);
    Save(file);
  }

  void Load(string fname) {
    ifstream file(fname, ios::binary);
    Load(file);
  }

  void Save(ofstream &file) {
    file << "certified_cosine_dynamic" << endl;
    file << vertexes.size() << " " << starting.size() << endl;
    for (uint i = 0; i < vertexes.size(); i++) {
      // this has to construct the number of neighbors and
      auto &v = vertexes[i];

      file << v.incoming_edges.size() << " " << v.outgoing_edges.size() << " " << v.proof_required << " ";
      // don't write floats as strings as it truncates the values and we are
      // unable to then use the value for constructing certificates.
      file.write((const char *)&v.proof_distance, sizeof(float_t));
      file.write((const char *)v.outgoing_edges.data(),
                 v.outgoing_edges.size() * sizeof(typename decltype(v.outgoing_edges)::value_type));
      file.write((const char *)v.incoming_edges.data(),
                 v.incoming_edges.size() * sizeof(typename decltype(v.incoming_edges)::value_type));
      file << endl;
    }
    file.write((const char *)starting.data(), sizeof(typename decltype(starting)::value_type) * starting.size());
  }

  void Load(ifstream &file) {
    string self_name;
    file >> self_name;
    assert(self_name == "certified_cosine_dynamic");
    uint nvertices, nstarting;
    file >> nvertices >> nstarting;
    vertexes.resize(nvertices);
    starting.resize(nstarting);
    for (int i = 0; i < nvertices; i++) {
      int incoming, outgoing, proof_required;
      float_t proof_v;
      auto &v = vertexes[i];
      file >> incoming >> outgoing >> proof_required;
      assert(proof_distance_set_size == -1 || proof_distance_set_size == outgoing);
      proof_distance_set_size = outgoing;
      char zz = file.get();
      assert(zz == ' ');
      (void)zz;
      file.read((char *)&proof_v, sizeof(float_t));
      v.outgoing_edges.resize(outgoing);
      file.read((char *)v.outgoing_edges.data(), outgoing * sizeof(typename decltype(v.outgoing_edges)::value_type));
      v.incoming_edges.resize(incoming);
      file.read((char *)v.incoming_edges.data(), incoming * sizeof(typename decltype(v.incoming_edges)::value_type));

      v.proof_required = proof_required;
      v.proof_distance = proof_v;

      get_vertex(i).build_all_edges(this);
    }
    char zz = file.get();
    assert(zz == '\n');
    (void)zz;
    file.read((char *)starting.data(), sizeof(typename decltype(starting)::value_type) * starting.size());
  }
};

}  // namespace certified_cosine

#endif
