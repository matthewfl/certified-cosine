import argparse
import time
import json
import sys

import numpy as np
import h5py

import certified_cosine



def main():
    argp = argparse.ArgumentParser()
    argp.add_argument('--input_h5', type=str, required=True)
    argp.add_argument('--save_edges', type=str)
    argp.add_argument('--load_edges', type=str)
    argp.add_argument('--output_results', type=str, required=True)
    argp.add_argument('--num_edges', type=int, default=50)
    argp.add_argument('--limit_expand', type=int)
    argp.add_argument('--version', action='version', version=certified_cosine.__version__)

    args = argp.parse_args()

    h5 = h5py.File(args.input_h5)
    train_vectors = h5['train'][:]

    # some of these datasets might have zero vectors.  Ideally, we would just
    # remove those vectors, however that would mean we would have to remap the
    # indexes etc and would technically have a smaller dataset.  instead we are
    # just going to set these to a "dummy" location and hope they don't hurt us
    # too much in the evaluation.  Really, these should not even be included in
    # the testing/training dataset as they are illdefined wrt to the angle and
    # are disinteresting in a euclidean setting
    train_vectors[np.where(np.linalg.norm(train_vectors, axis=1) == 0), 0] = 1

    train_vectors /= np.linalg.norm(train_vectors, axis=1)[:, np.newaxis]


    test_vectors = h5['test'][:]
    test_vectors[np.where(np.linalg.norm(test_vectors, axis=1) == 0), 0] = 1
    test_vectors /= np.linalg.norm(test_vectors, axis=1)[:, np.newaxis]


    if args.load_edges:
        processed = certified_cosine.open(train_vectors, args.load_edges)
    else:
        processed = certified_cosine.build(train_vectors, args.num_edges)

    if args.save_edges:
        processed.save(args.save_edges)

    engine = processed.engine(train_vectors)

    results = []

    expand_limit = train_vectors.shape[0]
    if args.limit_expand:
        expand_limit = args.limit_expand

    try:

        for ti in range(test_vectors.shape[0]):
            vec = test_vectors[ti]

            start = time.time()
            rr = engine.lookup_k_limit(vec, 1, expand_limit)
            end = time.time()

            [best], count, located_at, got_proof = rr

            results.append([bool(best == h5['neighbors'][ti, 0]), float(h5['distances'][ti, 0]), int(count), int(located_at), float(end - start)])
            sys.stdout.write(f'\r{ti}  ')
    finally:

        with open(args.output_results, 'w+') as f:
            json.dump(results, f, indent=1)




if __name__ == '__main__':
    main()
