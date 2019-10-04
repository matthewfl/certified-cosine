
/**
 * Written By Matthew Francis-Landau (2019)
 *
 * Python wrapper for certified cosine
 */

#include <fstream>
#include <utility>

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "pre_processing.hpp"
#include "storage.hpp"

// #include "word_vecs.hpp"

#include "lookup.hpp"
#include "policy.hpp"

// dimensions where having a specialized version for would be beneficial.  This
// can allow for inner products that are computed to be fully unrolled.
#define EXPAND_DIM(X) \
  X(25)               \
  X(50)               \
  X(100)              \
  X(200)              \
  X(300)              \
  X(64)               \
  X(128)              \
  X(256)              \
  X(512)

namespace py = pybind11;
using namespace std;

using namespace certified_cosine;
using namespace Eigen;

class cc_exception : std::exception {
 private:
  std::string str;
  const char *error;

 public:
  cc_exception(std::string s) : str(s), error(str.c_str()) {}
  cc_exception(const char *error) : error(error) {}
  const char *what() const noexcept override { return error; }
};

class WordVecs {
 public:
  Matrix<float, Dynamic, Dynamic, RowMajor> words;
  int words_cnt;
  int length;

  WordVecs(string fname) {
    ifstream file(fname);
    string word;
    file >> words_cnt;
    file >> length;

    float *buffer = new float[length];
    int i = 0;
    words = MatrixXf(words_cnt, length);  // rows, cols
    while (file >> word) {
      file.get();  // ignore space
      file.read((char *)buffer, sizeof(float) * length);
      for (int j = 0; j < length; j++) {
        words(i, j) = buffer[j];
      }
      i++;
      if (i >= words_cnt) break;
    }
    delete[] buffer;
  }
};

template <typename engine>
struct FastVectorsPythonWrapper {
  engine lookup_engine;

  py::object matrix_handle;
  py::object storage_handle;

  FastVectorsPythonWrapper(engine &eg) : lookup_engine(eg) {}
};

template <typename storage_t>
auto storage_wrapper(pybind11::module &m) {
  string name = "_Storage_";
  name += typeid(storage_t).name();

  auto ret =
      py::class_<storage_t>(m, name.c_str())
          .def(py::init<>())
          .def("save", [](storage_t &self, string fname) { self.Save(fname); })
          .def("load", [](storage_t &self, string fname) { self.Load(fname); })
          .def("size", [](storage_t &self) { return self.size(); })
          .def("neighbors",
               [](storage_t &self, int id) {
                 auto vertex = self.get_vertex(id);
                 auto opaque = vertex.neighbor_opaque(&self);
                 vector<int> ret;
                 for (int i = 0; i < vertex.size(&self); i++) {
                   ret.push_back(vertex.neighbor(&self, opaque, i));
                 }
                 return ret;
               })
          .def("num_neighbors",
               [](storage_t &self, int id) {
                 auto vertex = self.get_vertex(id);
                 return vertex.size(&self);
               })
          .def("proof_distance",
               [](storage_t &self, int id) {
                 auto vertex = self.get_vertex(id);
                 return vertex.proof_distance(&self);
               })

          .def("lookup_simple",
               [](storage_t &self, const PMatrix<float_t> &matrix,
                  Eigen::Ref<const Eigen::Matrix<float_t, Eigen::Dynamic, 1>> &vector) {
                 // perform a single lookup for a vector
                 // this does not hold onto the allocations (that should be faster)
                 // so this should not be the general method
                 //
                 // this should go through the lookup_wrapper, so that we can have more constrol
                 // over the optimizations of the lookup procedure

                 LookupCertifiedCosine l(matrix, &self);

                 OneBestPolicy<float_t> policy;
                 l.lookup(vector, policy);

                 return policy.id;
               })

          .def("lookup_simple_k",
               [](storage_t &self, const PMatrix<float_t> matrix,
                  Eigen::Ref<const Eigen::Matrix<float_t, Eigen::Dynamic, 1>> vector, int k) {
                 LookupCertifiedCosine l(matrix, &self);

                 NBestPolicy<float_t> policy(k);
                 l.lookup(vector, policy);

                 std::vector<int> ret;
                 while (!policy.items.is_empty()) {
                   ret.push_back(policy.items.max().id);
                   policy.items.remove_max();
                 }

                 return ret;
               })

          .def("engine", [](py::object self, py::object matrix) -> py::object {
            storage_t *self_s = self.cast<storage_t *>();

            typedef Eigen::Ref<const Eigen::Matrix<float_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>> m_0;

            // this is the matrix reference type with dynamic dimention we are
            // going to have to recast this in the case that the dimention
            // matches something we have special cased
            m_0 mat = matrix.cast<m_0>();

            int cols = mat.cols();
            if ((((intptr_t)mat.data()) & (16 - 1)) != 0) {
              // then the data is not aligned, so we can not use the more optimized version of the code
              std::cerr << "Warning: matrix is not aligned to 16 byte boundary, falling back to general engine "
                        << (((intptr_t)mat.data()) & (16 - 1)) << std::endl;
              cols = -1;
            }

            switch (cols) {
      // TODO: there is something strange when using the aligned16 and the pointer becomes invalid
#define ENGINE_C(C)                                                                                 \
  case C: {                                                                                         \
    typedef Eigen::Ref<const Eigen::Matrix<float_t, Eigen::Dynamic, C /* <-- use hard code size */, \
                                           Eigen::RowMajor> /* , Eigen::Aligned16*/>                \
        m_##C;                                                                                      \
    m_##C mat##C = mat;                                                                             \
    LookupCertifiedCosine<storage_t, m_##C> ee(mat##C, self_s);                                     \
    FastVectorsPythonWrapper rr(ee);                                                                \
    rr.matrix_handle = matrix;                                                                      \
    rr.storage_handle = self;                                                                       \
    return py::cast(rr);                                                                            \
  }
              EXPAND_DIM(ENGINE_C)
#undef ENGINE_C
              default: {
                LookupCertifiedCosine<storage_t, m_0> ee(mat, self_s);
                FastVectorsPythonWrapper rr(ee);
                rr.matrix_handle = matrix;
                rr.storage_handle = self;
                return py::cast(rr);
              }
            }
          });

  ;
  return ret;
}

template <typename engine_t, typename policy_t>
py::object lookup_k_limit(FastVectorsPythonWrapper<engine_t> &self,
                          const Eigen::Ref<const typename engine_t::VecD> &vector, int k, int limit) {
  py::gil_scoped_release release;

  // LimitExpand<CountingNBestPolicy<float_t>> policy(limit, k);
  policy_t policy(limit, k);

  self.lookup_engine.lookup(vector, policy);

  std::vector<int> ret;
  while (!policy.items.is_empty()) {
    ret.push_back(policy.items.max().id);
    policy.items.remove_max();
  }

  return py::make_tuple(ret, policy.count, policy.count_located, policy.got_proof());
}

template <typename engine_t>
auto lookup_engine_wrapper(pybind11::module &m) {
  string lname = "_LookupEngineWrapper_";
  lname += typeid(FastVectorsPythonWrapper<engine_t>).name();
  return py::class_<FastVectorsPythonWrapper<engine_t>>(m, lname.c_str())
      .def("lookup",
           [](FastVectorsPythonWrapper<engine_t> &self, const Eigen::Ref<const typename engine_t::VecD> &vector) {
             py::gil_scoped_release release;
             OneBestPolicy<float_t> policy;
             self.lookup_engine.lookup(vector, policy);
             return policy.id;
           })

      .def("lookup_k",
           [](FastVectorsPythonWrapper<engine_t> &self, const Eigen::Ref<const typename engine_t::VecD> &vector,
              uint k) {
             py::gil_scoped_release release;

             NBestPolicy<float_t> policy(k);

             self.lookup_engine.lookup(vector, policy);

             std::vector<int> ret;
             while (!policy.items.is_empty()) {
               ret.push_back(policy.items.max().id);
               policy.items.remove_max();
             }
             return ret;
           })

      .def("lookup_k_limit", lookup_k_limit<engine_t, LimitExpand<CountingNBestPolicy<float_t>>>, py::arg("vector"),
           py::arg("k"), py::arg("limit") = (((int)1) << 30))

      .def("lookup_k_limit_prove1", lookup_k_limit<engine_t, ProveBest<LimitExpand<CountingNBestPolicy<float_t>>>>,
           py::arg("vector"), py::arg("k"), py::arg("limit") = (((int)1) << 30))

      .def("lookup_k_limit_apxProve",
           lookup_k_limit<engine_t, ApproximatePolicy<LimitExpand<CountingNBestPolicy<float_t>>>>, py::arg("vector"),
           py::arg("k"), py::arg("limit") = (((int)1) << 30))

      .def_property_readonly_static("static_columns",
                                    [](py::object &self) -> int { return engine_t::MatD::ColsAtCompileTime; })

      ;
}

template <typename float_t>
void define_module(pybind11::module &m) {
  m.def("open", [](py::array_t<float_t, py::array::c_style> &arr, const string fname) -> py::object {
    // for opening an already processed file
    py::gil_scoped_release release;

    ifstream input(fname);
    string type;
    input >> type;
    input.clear();
    input.seekg(0, ios::beg);
    if (type == "fast_vectors_compact") {
      compact_storage<float_t> *ret = new compact_storage<float_t>;
      ret->Load(input);
      return py::cast(ret);
    } else if (type == "fast_vectors_dynamic") {
      dynamic_storage<float_t> *ret = new dynamic_storage<float_t>;
      ret->Load(input);
      return py::cast(ret);
    } else {
      throw std::runtime_error("Unrecognized file header");
    }
  });

  lookup_engine_wrapper<
      LookupCertifiedCosine<dynamic_storage<float_t>,
                            Eigen::Ref<const Eigen::Matrix<float_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>>>(
      m);

  lookup_engine_wrapper<
      LookupCertifiedCosine<compact_storage<float_t>,
                            Eigen::Ref<const Eigen::Matrix<float_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>>>(
      m);

  // expand versions specific to various dimentions
#define ENGINE_C(C)                                                                                             \
  lookup_engine_wrapper<LookupCertifiedCosine<                                                                  \
      dynamic_storage<float_t>,                                                                                 \
      Eigen::Ref<const Eigen::Matrix<float_t, Eigen::Dynamic, C, Eigen::RowMajor> /*, Eigen::Aligned16*/>>>(m); \
  lookup_engine_wrapper<LookupCertifiedCosine<                                                                  \
      compact_storage<float_t>,                                                                                 \
      Eigen::Ref<const Eigen::Matrix<float_t, Eigen::Dynamic, C, Eigen::RowMajor> /*, Eigen::Aligned16*/>>>(m);

  EXPAND_DIM(ENGINE_C)
#undef ENGINE_C

  storage_wrapper<dynamic_storage<float_t>>(m)
      .def("make_compact",
           [](dynamic_storage<float_t> &self) {
             unique_ptr<compact_storage<float_t>> ret(new compact_storage<float_t>);
             self.BuildCompactStorage(*ret.get());
             return ret;
           })
      .def("neighbors_dists",
           [](dynamic_storage<float_t> &self, int id) {
             vector<py::tuple> ret;
             auto vertex = self.get_vertex(id);
             for (auto e : vertex.get_all_edges(&self)) {
               ret.push_back(py::make_tuple(e.id, e.score));
             }
             return ret;
           })
      .def(
          "make_smaller",
          [](dynamic_storage<float_t> &self, const PMatrix<float_t> &matrix, int new_size, int num_starting_points) {
            py::gil_scoped_release release;

            dynamic_storage<float_t> new_storage;
            make_smaller_all(matrix, self, new_storage, new_size, num_starting_points);
            return new_storage;
          },
          py::arg("matrix"), py::arg("new_size"), py::arg("num_starting_points") = (1 << 16));
  storage_wrapper<compact_storage<float_t>>(m);

  m.def("preprocess", preprocess<float_t>, py::call_guard<py::gil_scoped_release>());
  m.def("preprocess_exact_neighbors", exact_neighbors<float_t>, py::call_guard<py::gil_scoped_release>());
  m.def("preprocess_reverse_edges", reverse_edges<float_t>, py::call_guard<py::gil_scoped_release>());
  m.def("preprocess_build_all_edges", build_all_edges<float_t>, py::call_guard<py::gil_scoped_release>());
  m.def("preprocess_starting_approximation", starting_approximations<float_t>,
        py::call_guard<py::gil_scoped_release>());
  m.def("preprocess_shuffle_all_edges", shuffle_all_edges<float_t>, py::call_guard<py::gil_scoped_release>());

  m.def(
      "build",
      // TODO: this needs to take the eigen reference type as this may require copying the matrix
      [](const PMatrix<float_t> &matrix, uint num_neighbors, uint starting_points) {
        py::gil_scoped_release release;

        unique_ptr<dynamic_storage<float_t>> ret(new dynamic_storage<float_t>);
        if (__builtin_popcount(starting_points) != 1) {
          throw cc_exception("starting points should be a power of 2");
        }
        if (num_neighbors < 5) {  // probably a lot higher than 5
          throw cc_exception("num neighbors needs to be higher");
        }

        // check that the matrix has norm 1 for all of the elements
        if (!matrix.rowwise().norm().isApproxToConstant(1)) {
          throw cc_exception("matrix must already have norm 1 for all elements");
        }

        preprocess(matrix, *ret.get(), num_neighbors, starting_points);

        return ret;
      },
      py::arg("matrix"), py::arg("num_neighbors") = 50, py::arg("starting_points") = (1 << 16));
}

auto load_word_vectors(std::string fname) {
  // load word vectors in the binary format
  // this seems to have to copy the values out, though would like to move them
  WordVecs vecs(fname);
  return vecs.words;
}

PYBIND11_MODULE(certified_cosine, m) {
  m.doc() = "Fast nearest neighbor with certificates";

  define_module<float>(m);
  // define_module<double>(m);

  // this isn't really related to certified cosine, should be removed
  m.def("load_word_vectors", load_word_vectors, py::return_value_policy::move);

  static py::exception<cc_exception> exc(m, "CertifiedCosineError");
  py::register_exception_translator([](std::exception_ptr p) {
    try {
      if (p) std::rethrow_exception(p);
    } catch (const cc_exception &e) {
      exc(e.what());
    }
  });

#ifdef VERSION_INFO
  m.attr("__version__") = VERSION_INFO;
#else
  m.attr("__version__") = "dev";
#endif
}

// This is the only non-header file that we need.  Rather than dealing with
// having to link another compiled file, just including it here.
#include "pre_processing.cc"
