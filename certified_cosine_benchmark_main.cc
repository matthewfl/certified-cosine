#include <iostream>
#include <random>

#include <omp.h>

#include <Eigen/Dense>

#define CERTIFIEDCOSINE_DEBUG_ACCESS
#define CERTIFIEDCOSINE_DEBUG

#include "lookup.hpp"
#include "policy.hpp"
#include "pre_processing.hpp"
#include "vector_stats.hpp"

using namespace std;
using namespace Eigen;

// #define RANDOM_TEST
// #define RANDOM_TEST_NDIM 30

// int global_number_edges = 50;
//#define NUMBER_EDGES (int)(global_number_edges)

//#define NUMBER_EDGES_RAND 9
//#define NUMBER_BOOTSTRAP 400
///#define NUMBER_RAND_LOOKS 3

// the batch size that we should split the vectors up per thread
//#define PARALLEL_BATCH_SIZE 100

// the number of items from the origional matrix that need to get multiplied by at a time
//#define PARALLEL_MATMUL_SIZE 10

/**
 * Read a file containing embeddings of word vectors in the binary format.
 */
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

template <typename matrix_t, typename storage_t>
void random_test(const matrix_t &matrix, const storage_t &storage, ostream &file) {
  file << "test,"
          "success,"
          "proof,"
          "around_area,"
          "best_distance,"
          "selected_distance,"
          "proof_distance,"
          "number_inner_products,"
          "number_till_located,"
          "x_steps,"
          "num_prove_items,"
          "optimize_calls,"
          "optimize_loops,"
          "time(ms),"
          "direct_time(ms)\n";

  int lcnt = 0;

  using namespace certified_cosine;

  typedef Eigen::Matrix<typename matrix_t::Scalar, matrix_t::ColsAtCompileTime, 1> VecD;
  typedef matrix_t MatD;

  auto randomVector = [&]() { return VecD::Random(matrix.cols()).eval().normalized().eval(); };

#ifdef CERTIFIEDCOSINE_USE_PARALLEL
#pragma omp parallel
#endif
  {
    // uint num_innerproducts = 0;
    // auto ignore = [&](int a) {
    //                 num_innerproducts++;
    //                 // there is nothing to ignore
    //                 return true;
    //               };
    // auto stop = [](int a) { return false; }; // don't stop looking for vertices
    // // use the proof to identify what it is locating
    // lookupManager_2<decltype(ignore), decltype(stop), true> lookupMan(ignore, stop, *this);

    LookupCertifiedCosine<storage_t, Eigen::Ref<const MatD>> lookupMan(matrix, &storage);

    struct timespec start_time, end_time;

#ifdef CERTIFIEDCOSINE_USE_PARALLEL
#pragma omp for schedule(dynamic, 3)
#endif
    for (int i = 0; i < 120000; i++) {
      // num_innerproducts = 0;
      // lookupMan.reset();

      int t = rand() % matrix.rows();
      VecD vec;
      if (i < 60000) {
        vec = (matrix.row(t).transpose() + ((i / 3) / 5000.) * randomVector()).normalized().eval();
      } else {
        vec = randomVector().normalized().eval();
      }

      // do the brute force method first, so in the debugger can know what the maxIndex is for comparison
      clock_gettime(CLOCK_MONOTONIC_RAW, &start_time);
      typename MatD::Index maxIndex;
      float_t best_score = (matrix * vec).maxCoeff(&maxIndex);
      clock_gettime(CLOCK_MONOTONIC_RAW, &end_time);

      float delta_ms2 =
          ((end_time.tv_sec - start_time.tv_sec) * 1000000 + (end_time.tv_nsec - start_time.tv_nsec) / 1000) / 1000.0;

      // CountExpandPolicy<OneBestPolicy<typename matrix_t::Scalar>> policy;
      CountingTillBest<typename matrix_t::Scalar> policy;
      // CountingNBestPolicy<typename matrix_t::Scalar> policy(2);

      clock_gettime(CLOCK_MONOTONIC_RAW, &start_time);
      // lookupMan.lookupOperation(vec, 25, 0);
      lookupMan.lookup(vec, policy);
      clock_gettime(CLOCK_MONOTONIC_RAW, &end_time);

      int selected = policy.id;                  // policy.items.max().id;
      float_t selected_score = policy.distance;  // policy.items.max().score;

      float delta_ms1 =
          ((end_time.tv_sec - start_time.tv_sec) * 1000000 + (end_time.tv_nsec - start_time.tv_nsec) / 1000) / 1000.0;

      // the number of vertices that could move the current proof item if they were located
      int around_area = 0;
      if (lookupMan.lp_project.located_cosim > policy.proof_distance()) {
        // then we are going to search for more items that match the work with the inner product
        auto scores = (matrix * lookupMan.target.row(0).transpose()).array().eval();
        for (int i = 0; i < matrix.rows(); i++) {
          auto vx = storage.get_vertex(i);
          if (compareScore(scores(i), vx.proof_distance(&storage))) {
            around_area++;
          }
        }
      }

      assert(selected == maxIndex);

#ifdef CERTIFIEDCOSINE_USE_PARALLEL
#pragma omp critical
#endif
      {
        file << (lcnt++) << "," << (selected == maxIndex) << "," << (int)policy.got_proof() << "," << around_area << ","
             << best_score << "," << selected_score << "," << lookupMan.located_cosim << "," << policy.count << ","
             << policy.count_located << "," << lookupMan.score_step << "," << lookupMan.lp_project.num_rows << ","
             << lookupMan.lp_project.optimize_calls << "," << lookupMan.lp_project.optimize_loops << "," << delta_ms1
             << "," << delta_ms2 << endl;

        // assert that if we are finding a proof that we were actually successful
        // if(lookupMan.found_proof && selected != maxIndex) {
        //   cout << "======================================= FAILED " << (lcnt - 1) << "
        //   ================================\n";
        //   //assert(selected == maxIndex);
        // }
      }
    }
  }
}

// method in svd_wrapper.cc as it takes a while to compile
MatrixXf getSVD_U(const MatrixXf &inp);
// {
//   return inp.bdcSvd(Eigen::ComputeFullU).matrixU();
// }

int main(int argc, char **argv) {
  Eigen::initParallel();

  int num_neighbors = 50;  // number of outgoing edges to run with
  string input_filename;
  string output_edges_filename;
  string lookup_log_filename;
  string load_edges;

  int num_random_vectors = 50000;
  int embedding_dim = 30;
  int random_seed = 0;

  int use_cuda = 0;

  int use_compact = 0;

  int svd_reduce = 0;  // if this should try and reduce the dim using the svd

  for (int c = 1; c < argc; c++) {
    string s = argv[c];
    if (s == "--outgoing") {
      num_neighbors = atoi(argv[c + 1]);
      c++;
    } else if (s == "--input") {
      input_filename = argv[c + 1];
      c++;
      num_random_vectors = 0;
    } else if (s == "--save_edges") {
      output_edges_filename = argv[c + 1];
      c++;
    } else if (s == "--test_vecs") {
      num_random_vectors = atoi(argv[c + 1]);
      c++;
    } else if (s == "--dim") {
      embedding_dim = atoi(argv[c + 1]);
      c++;
    } else if (s == "--svd") {
      svd_reduce = 1;
      // no c++; as there are no additional arguments
    } else if (s == "--load_edges") {
      load_edges = argv[c + 1];
      c++;
    } else if (s == "--seed") {
      random_seed = atoi(argv[c + 1]);
      c++;
    } else if (s == "--log") {
      lookup_log_filename = argv[c + 1];
      c++;
    } else if (s == "--compact") {
      use_compact = 1;

      // } else if(s == "--cuda") {
      // #ifdef USE_CUDA
      //       use_cuda = 1;
      // #else
      //       cerr << "not compiled with cuda\n";
      //       return 1;
      // #endif
    } else if (s == "--help") {
      cout << argv[0]
           << ": A testing program for certified cosine\n"
              "Dataset can either be word vectors in the binary format or randomly generated\n"
              "\n"
              "NOTE !!!:  This compares against brute force linear scan, and thus is much slower than just running "
              "certified cosine\n"
              "\n"
              "    --outgoing    Number of outgoing edges from every vertex in the neighbor graph [default: 50]\n"
              "    --input       Input file name of binary word embedding vectors to test [default: Not set, randomly "
              "generate dataset]\n"
              "    --save_edges  Save the KNNG neighborhood graph [default: Not set]\n"
              "    --load_edges  Load the KNNG neighborhood graph instead of constructing it [default: Not set]\n"
              "    --dim         Dimension of randomly generated vectors or reduce rank of loaded vectors [default: "
              "30]\n"
              "    --svd         If set, reduce the dimension of loaded word embeddings otherwise use datasets "
              "dimension [default: Not set]\n"
              "    --seed        Random seed for generating random dataset [default: 0]\n"
              "    --log         CSV file to log the results of running [default: stdout]\n"
              "    --compact     Use the compacted storage when benchmarking queries\n";
      return 1;
    } else {
      cerr << "failed to parse argument " << s << endl;
      return 1;
    }
  }

  // #ifdef USE_CUDA
  //   // claim the gpu early
  //   if(use_cuda) {
  //     cuda_dummy_init();
  //   }
  // #endif

  // global_number_edges = num_neighbors;

  typedef Matrix<float, Dynamic, Dynamic, RowMajor> Mat;
  Mat embeddedMatrix;

  if (input_filename.empty()) {
    // then create a random representation for this
    normal_distribution<float> nd;
    mt19937 rng(random_seed);

    auto testMat =
        MatrixXf::Zero(num_random_vectors, embedding_dim).unaryExpr([&](auto f) -> float { return nd(rng); }).eval();

    auto norms = testMat.rowwise().norm().eval();
    embeddedMatrix = testMat.array().colwise() / norms.array();
  } else {
    // load word embedding file, this will be about a 3 million if this is google word vectors
    // this also will need the embedding dimention to match

    WordVecs wv(input_filename);

    if (svd_reduce) {
      // take the svd of the word vectors, and then use the top embedding_dim rows of the matrix
      // this way, we will be able to use the real data but have a slightly faster operation when it comes to
      // experimenting with stuff

      // compiling the svd is really slow, so maybe just leave that out for now?
      // embeddedMatrix = wv.words;
      // auto a = getVSVD(embeddedMatrix);
      // auto a = wv.words.bdcSvd(Eigen::ComputeFullV).matrixV();
      // JacobiSVD<MatrixXf> svd(wv.words, ComputeThinU | ComputeThinV);
      // embeddedMatrix = svd.matrixU().leftCols(embedding_dim);
      auto U = getSVD_U(wv.words.transpose()).leftCols(embedding_dim);
      embeddedMatrix = wv.words * U;
      auto norms = embeddedMatrix.rowwise().norm().eval();
      embeddedMatrix.array().colwise() /= norms.array();
    } else {
      embeddedMatrix = wv.words;
      // normalize the vectors
      auto norms = embeddedMatrix.rowwise().norm().eval();
      embeddedMatrix.array().colwise() /= norms.array();
    }
  }

  // #ifndef NDEBUG
  //   if(0) {
  //     // this should automatically check that the dimentions align correctly
  //     // since we have hard coded in what the shapes of things are
  //     auto tm = Matrix<float, 11, 30>::Random();
  //     FastLookup14<decltype(tm)> gg(tm);
  //   }
  // #endif

  using namespace certified_cosine;
  dynamic_storage<float> storage;

  // exact_neighbors(embeddedMatrix, storage, num_neighbors);
  if (load_edges.empty()) {
    preprocess<float>(embeddedMatrix, storage, num_neighbors);
  } else {
    storage.Load(load_edges);
  }

  print_summarization(storage);

  if (!output_edges_filename.empty()) {
    ofstream save_file(output_edges_filename);
    storage.Save(save_file);
  }

  if (use_compact) {
    // compact storage
    compact_storage<float> cstorage;
    storage.BuildCompactStorage(cstorage);

    if (!lookup_log_filename.empty()) {
      ofstream log_file(lookup_log_filename);
      random_test(embeddedMatrix, cstorage, log_file);
    } else {
      random_test(embeddedMatrix, cstorage, cout);
    }
  } else {
    // dynamic storage

    if (!lookup_log_filename.empty()) {
      ofstream log_file(lookup_log_filename);
      random_test(embeddedMatrix, storage, log_file);
    } else {
      random_test(embeddedMatrix, storage, cout);
    }
  }

  return 0;
}
