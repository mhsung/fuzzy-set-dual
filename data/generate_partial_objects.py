#!/usr/bin/env python
# Minhyuk Sung (mhsung@cs.stanford.edu)
# March 2018

import os, sys
BASE_DIR = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(BASE_DIR, '..'))
sys.path.append(os.path.join(BASE_DIR, '..', 'fuzzy_set_dual', 'network'))
sys.path.append(os.path.join(BASE_DIR, '..', 'network_utils'))

from global_variables import *
from dataset import Dataset
from resample_points import resample_points, centerize_points
import itertools
import numpy as np
import random


def generate_all_subgraphs(test_data):
    point_clouds = None
    centered_point_clouds = None
    centers = None
    component_idxs = []
    complement_idxs = np.array([], dtype=int)

    for graph_idx in range(test_data.n_data):
        nodes = list(test_data.comp_graphs[graph_idx].nodes)
        subsets = test_data.comp_all_subsets[graph_idx]
        n_subsets = len(subsets)

        for i, i_idxs in enumerate(subsets):
            P = resample_points(test_data.orig_points[i_idxs],
                    test_data.areas[i_idxs])
            centered_P, center = centerize_points(P)

            P = np.expand_dims(P, 0)
            point_clouds = P if point_clouds is None \
                    else np.vstack((point_clouds, P))

            centered_P = np.expand_dims(centered_P, 0)
            centered_point_clouds = centered_P \
                    if centered_point_clouds is None \
                    else np.vstack((centered_point_clouds, centered_P))

            center = np.expand_dims(center, 0)
            centers = center if centers is None \
                    else np.vstack((centers, center))

            component_idxs.append(i_idxs)

        graph_complement_idxs = np.full(n_subsets, -np.inf, dtype=int)
        for (i, j) in itertools.combinations(range(n_subsets), 2):
            if graph_complement_idxs[i] >= 0 or \
                    graph_complement_idxs[j] >= 0: continue
            i_idxs = subsets[i]
            j_idxs = subsets[j]
            i_complement_idxs = [x for x in nodes if x not in i_idxs]
            if i_complement_idxs == j_idxs:
                graph_complement_idxs[i] = j
                graph_complement_idxs[j] = i

        graph_complement_idxs += complement_idxs.size
        complement_idxs = np.concatenate((
            complement_idxs, graph_complement_idxs))

        if (graph_idx + 1) % 10 == 0:
            print('{:d}/{:d} processed...'.format(graph_idx + 1,
                test_data.n_data))

    n_all_subsets = point_clouds.shape[0]
    assert(len(component_idxs) == n_all_subsets)
    assert(complement_idxs.size == n_all_subsets)
    print('# all subsets: {:d}'.format(n_all_subsets))


    np.save(g_partial_test_point_clouds_file, point_clouds)
    print("Saved '{}'.".format(g_partial_test_point_clouds_file))

    np.save(g_partial_test_centered_point_clouds_file, centered_point_clouds)
    print("Saved '{}'.".format(g_partial_test_centered_point_clouds_file))

    np.save(g_partial_test_positions_file, centers)
    print("Saved '{}'.".format(g_partial_test_positions_file))

    with open(g_partial_test_component_idxs_file, 'w') as f:
        for idxs in component_idxs:
            f.write(' '.join([str(x) for x in idxs]) + '\n')
    print("Saved '{}'.".format(g_partial_test_component_idxs_file))

    np.savetxt(g_partial_test_complement_idxs_file, complement_idxs, fmt='%d')
    print("Saved '{}'.".format(g_partial_test_complement_idxs_file))


if __name__ == '__main__':
    np.random.seed(0)
    random.seed(0)

    test_data = Dataset(
            g_component_test_pairs_file,
            g_component_all_component_labels_file,
            g_component_all_centered_point_clouds_file,
            g_component_all_positions_file,
            g_component_all_areas_file,
            batch_size=32,
            centerize=False)

    if not os.path.exists(g_partial_dir):
        os.makedirs(g_partial_dir)

    generate_all_subgraphs(test_data)

