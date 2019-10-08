# Certified Cosine (![](http://latex.codecogs.com/gif.latex?%5Cmathbf%7BC%7D_2))

Certified Cosine is a fast method for generating certificates for nearest
neighbor methods in high dimensional spaces.  Certified Cosine is not
probabilistic but rather *certifies* that the returned result is correct.

More information can be found in our paper [here](https://arxiv.org/abs/1910.02478)


## Usage (Python)

An example is contain in [runtests.py](./runtests.py).

```python
import certified_cosine

# load 10,000 vectors of dimention 100 into memory
vectors = np.random(10000, 100)

# norm the vectors such that the inner product between two vectors is the cosine similarity
vectors /= np.linalg.norm(vectors, axis=1)[:, np.newaxis]


cc = certified_cosine.build(vectors)

# OPTIONAL: save the KNNG graph for use later
cc.save("my_file")

# OPTIONAL: load the KNNG graph /instead/ of building it new everytime
cc = certified_cosine.open(vectors, "my_file")



# RECOMMENDED: create a lookup engine to reuse internal search data structures
engine = cc.engine(vectors)



# perform a lookup for the 10-nearest neighbors
query = np.random.rand(100)     # the query vector.
query /= np.linalg.norm(query)


 # arguments:
 #   Query: the vector that we are looking for its nearest neighbor
 #   K: The number of neighbors to return
 #   limit: Budget before returning the current best guess.
 # Return values:
 #   Neighbors: A list of the integer ids for the k-top nearest neighbors
 #   Number expanded: The amount of the budget that was consumed
 #   Created Certificate: True if the certificate was created, false if the budget stops search early.

neighbors, number_expanded, _, created_certificate = engine.lookup_k_limit(query, 10, 50000)


```

## Usage (C++)


```CPP
#include <certified_cosine/lookup.hpp>
#include <certified_cosine/preprocesses.hpp>
#include <certified_cosine/storage.hpp>
#include <certified_cosine/policy.hpp>

// certified cosine depends on eigen for its matrix
#include <Eigen/Dense>


// construct a random vector matrix
Eigen::MatrixXf vectors = MatrixXf::Random(10000, 100);
vectors.array().colwise() /= vectors.rowwise().norm();  // norm the vectors


// construct the KNNG during preprocessing
certified_cosine::dynamic_storage<float> storage;
certified_cosine::preprocess<float>(vectors, storage, 50 /* number neighbors */);

// OPTIONAL: to load or save the KNNG graph
storage.Save("my_file");
stroage.Load("my_file");


// construct a lookup engine to perform queries against.
// Can be reused to avoid reallocation.
// For handling multiple threads, multiple engines can be craeted (one per thread)
certified_cosine::lookup_proof<decltype(storage), Eigen::Ref<const decltype(vectors)>> engine(vectors, &storage);

// a query vector
Eigen::VectorXf query = Eigen::VectorXf::Random(100).normalized();

// policy controls how the lookup is performed, number of neighbors looking for
// and how certification interacts.  Here set top-10 nearest neighbors with a budget of 50,000
certified_cosine::LimitExpand<certified_cosine::CountingNBestPolicy<float>> policy(50000, 10);






```



## Install instructions


For python:
```bash
pip install 'git+https://github.com/matthewfl/certified-cosine.git'

Or:
git clone https://github.com/matthewfl/certified-cosine.git
cd certified-cosine
pip install .
```


This has been tested with g++-7 and up
This requires [Eigen](http://eigen.tuxfamily.org/).




## Cite

```
@article{certified_cosine,
 author = {Francis-Landau, Matthew and Van Durme, Benjamin}
 title = {Fast and/or Exact Nearest Neighbors}
 year = {2019}
 url = {https://arxiv.org/abs/1910.02478}
 keywords={Nearest Neighbors,Certificates,Cosine similarity}
}

```
