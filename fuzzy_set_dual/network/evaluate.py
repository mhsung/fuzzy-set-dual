# Minhyuk Sung (mhsung@cs.stanford.edu)
# March 2018

import os, sys
BASE_DIR = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(BASE_DIR, '..', '..'))
sys.path.append(os.path.join(BASE_DIR, '..', '..', 'network_utils'))

from global_variables import *
from dataset import *
from generate_outputs import *
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata
import numpy as np
import pandas as pd


kTop = 10


def retrieve_and_evaluate_complements(sess, net, data, centerize, out_dir):
    if not os.path.exists(out_dir): os.makedirs(out_dir)

    if centerize:
        X = np.load(g_partial_test_centered_point_clouds_file)
    else:
        X = np.load(g_partial_test_point_clouds_file)
    n_all_subsets = X.shape[0]

    complement_idxs = np.genfromtxt(g_partial_test_complement_idxs_file,
            dtype=int)
    assert(len(complement_idxs) == n_all_subsets)

    print('Compute embedding coordinates...')
    fX = predict(X, sess, net, 'f')
    gX = predict(X, sess, net, 'g')
    print('Done.')

    print('Compute ranks...')
    retrievals = np.empty((n_all_subsets, kTop), dtype=int)
    ranks = np.empty(n_all_subsets, dtype=int)
    percentile_ranks = np.empty(n_all_subsets)

    for i in range(n_all_subsets):
        E_XY = inclusion_errors(
                np.expand_dims(fX[i], 0),
                gX)
        E_YX = inclusion_errors(
                fX,
                np.expand_dims(gX[i], 0))
        E = E_XY + E_YX
        if (i + 1) % 100 == 0:
            print('{:d}/{:d} processed...'.format(i + 1, n_all_subsets))

        retrievals[i] = np.argsort(E)[:kTop]
        i_ranks = rankdata(E, method='min')
        ranks[i] = i_ranks[complement_idxs[i]]
        percentile_ranks[i] = float(np.sum(E >= E[complement_idxs[i]])) /\
                float(n_all_subsets) * 100.
    print('Done.')

    recall_1 = float(np.sum(ranks <= 1)) / n_all_subsets
    recall_10 = float(np.sum(ranks <= 10)) / n_all_subsets
    rank_med = np.median(ranks)
    rank_mean = np.mean(ranks)
    percentile_rank_med = np.median(percentile_ranks)
    percentile_rank_mean = np.mean(percentile_ranks)

    out = ''
    out += '# Partial Shapes: {:d}\n'.format(n_all_subsets)
    out += 'Recall@1: {:.1f}\n'.format(recall_1 * 100.)
    out += 'Recall@10: {:.1f}\n'.format(recall_10 * 100.)
    #out += 'Median: {:.1f}\n'.format(rank_med)
    #out += 'Mean: {:.1f}\n'.format(rank_mean)
    out += 'Median Percentile Rank: {:.1f}\n'.format(percentile_rank_med)
    out += 'Mean Percentile Rank: {:.1f}\n'.format(percentile_rank_mean)
    print(out)

    out_file = os.path.join(out_dir, 'complementarity_stats.txt')
    with open(out_file, 'w') as f: f.write(out)
    print("Saved '{}'.".format(out_file))

    out_file = os.path.join(out_dir, 'complementarity_retrievals.txt')
    np.savetxt(out_file, retrievals, fmt='%d', delimiter=' ')
    print("Saved '{}'.".format(out_file))


def retrieve_interchangeables(sess, net, data, centerize, out_dir):
    if not os.path.exists(out_dir): os.makedirs(out_dir)

    if centerize:
        X = np.load(g_partial_test_centered_point_clouds_file)
    else:
        X = np.load(g_partial_test_point_clouds_file)
    n_all_subsets = X.shape[0]

    print('Compute embedding coordinates...')
    fX = predict(X, sess, net, 'f')
    gX = predict(X, sess, net, 'g')
    print('Done.')

    print('Compute distances...')
    f_dists = np.linalg.norm(np.minimum(
        np.expand_dims(fX, 1), np.expand_dims(fX, 0)), axis=-1)
    g_dists = np.linalg.norm(np.maximum(
        np.expand_dims(gX, 1), np.expand_dims(gX, 0)), axis=-1)
    dists = g_dists - f_dists
    print('Done.')

    print('Retrieve neighbors...')
    retrievals = np.empty((n_all_subsets, kTop), dtype=int)
    for i in range(n_all_subsets):
        idxs = np.argsort(dists[i])
        idxs = idxs[idxs != i]
        retrievals[i] = idxs[:kTop]

    out_file = os.path.join(out_dir, 'interchangeability_retrievals.txt')
    np.savetxt(out_file, retrievals, fmt='%d', delimiter=' ')
    print("Saved '{}'.".format(out_file))


def compute_part_interchangeability(sess, net, data, centerize, out_dir):
    assert(centerize)

    assert(os.path.exists(g_part_all_part_labels_file))
    df = pd.read_csv(g_part_all_part_labels_file, index_col=False)
    md5s = df['md5'].tolist()
    labels = [int(x) for x in df['idx'].tolist()]
    labels = np.array(labels)
    n_parts = labels.size

    out_file = os.path.join(out_dir, 'part_labels.txt')
    np.savetxt(out_file, labels, fmt='%d')
    print("Saved '{}'.".format(out_file))

    X = np.load(g_part_all_centered_point_clouds_file)
    assert(X.shape[0] == n_parts)

    print('Compute embedding coordinates...')
    fX = predict(X, sess, net, 'f')
    gX = predict(X, sess, net, 'g')
    print('Done.')

    print('Compute distances...')
    f_dists = np.linalg.norm(np.minimum(
        np.expand_dims(fX, 1), np.expand_dims(fX, 0)), axis=-1)
    g_dists = np.linalg.norm(np.maximum(
        np.expand_dims(gX, 1), np.expand_dims(gX, 0)), axis=-1)
    dists = g_dists - f_dists

    out_file = os.path.join(out_dir, 'part_interchangeability_distances.npy')
    np.save(out_file, dists)
    print("Saved '{}'.".format(out_file))

