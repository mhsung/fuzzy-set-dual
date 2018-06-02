# Minhyuk Sung (mhsung@cs.stanford.edu)
# March 2018

import os, sys
BASE_DIR = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(BASE_DIR, '..', '..', 'network_utils'))

from itertools import chain, combinations
from resample_points import resample_points, centerize_points
import math
import networkx as nx
import numpy as np
import pandas as pd
import random


def powerset(iterable, exclude_entire_set=False):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    # https://docs.python.org/3/library/itertools.html#recipes
    # NOTE: Do not include empty set.
    s = list(iterable)
    if exclude_entire_set:
        ret = chain.from_iterable(combinations(s, r) for r in range(1,len(s)))
    else:
        ret = chain.from_iterable(combinations(s, r) for r in range(1,len(s)+1))
    return ret


def is_binary_split(G, nodes):
    rev_nodes = G.nodes - nodes
    if not rev_nodes: return False

    H = G.subgraph(nodes)
    H_rev = G.subgraph(rev_nodes)
    return nx.is_connected(H) and nx.is_connected(H_rev)


class Dataset(object):
    def __init__(self,
            pairs_file,
            all_component_labels_file,
            all_point_clouds_file,
            all_positions_file,
            all_areas_file,
            batch_size,
            centerize):
        self.batch_size = batch_size
        self.centerize = centerize

        assert(os.path.exists(all_component_labels_file))
        df = pd.read_csv(all_component_labels_file, index_col=False)
        self.all_md5s = df['md5'].tolist()
        self.all_comp_labels = [int(x) for x in df['idx'].tolist()]

        assert(os.path.exists(all_point_clouds_file))
        self.centered_points = np.load(all_point_clouds_file)
        self.n_components = self.centered_points.shape[0]
        self.n_points = self.centered_points.shape[1]
        assert(self.centered_points.shape[2] == 3)

        assert(os.path.exists(all_positions_file))
        self.positions = np.load(all_positions_file)
        assert(self.positions.shape[0] == self.n_components)
        assert(self.positions.shape[1] == 3)

        assert(os.path.exists(all_areas_file))
        self.areas = np.load(all_areas_file)
        assert(self.areas.size == self.n_components)

        # Compute point clouds in the original positions.
        self.orig_points = self.centered_points +\
                np.expand_dims(self.positions, axis=1)

        # Read pair list.
        df = pd.read_csv(pairs_file, index_col=False)
        unique_md5_list = df['md5'].unique().tolist()
        self.graph_md5s = []
        self.comp_graphs = []
        self.comp_all_subsets = []

        for md5 in unique_md5_list:
            G = nx.Graph()
            for idx in range(len(self.all_md5s)):
                if self.all_md5s[idx] == md5: G.add_node(idx)

            pairs = df[df['md5'] == md5]
            for _, row in pairs.iterrows():
                idx1 = row['idx1']
                idx2 = row['idx2']
                assert(G.has_node(idx1))
                assert(G.has_node(idx2))
                G.add_edge(idx1, idx2)

            # NOTE:
            # Exclude the entire set.
            all_subsets = [list(x) for x in powerset(G.nodes,
                exclude_entire_set=True) if is_binary_split(G, x)]
            if not all_subsets: continue

            self.graph_md5s.append(md5)
            self.comp_graphs.append(G)
            self.comp_all_subsets.append(all_subsets)

        self.n_data = len(self.comp_graphs)


    def generate_random_samples(self, graph_idxs):
        n_graphs = graph_idxs.size

        # Superset.
        X = np.empty((n_graphs, self.n_points, 3))

        # Positive example.
        Y = np.empty((n_graphs, self.n_points, 3))

        X_idxs_list = []
        Y_idxs_list = []

        for i in range(n_graphs):
            graph_idx = graph_idxs[i]
            G = self.comp_graphs[graph_idx]

            X_idxs = random.choice(self.comp_all_subsets[graph_idx])
            Y_idxs = [x for x in G.nodes if x not in X_idxs]
            assert(X_idxs)
            assert(Y_idxs)

            X_idxs_list.append(X_idxs)
            Y_idxs_list.append(Y_idxs)

            # Resample points in the all other components.
            X[i] = resample_points(self.orig_points[X_idxs], self.areas[X_idxs])
            Y[i] = resample_points(self.orig_points[Y_idxs], self.areas[Y_idxs])

            if self.centerize:
                X[i], _ = centerize_points(X[i])
                Y[i], _ = centerize_points(Y[i])

        return X, Y, X_idxs_list, Y_idxs_list


    def get_all_components(self):
        X = self.orig_points
        return X


    def get_data_objects(self):
        X = np.empty((self.n_data, self.n_points, 3))
        X_idxs_list = []
        X_memberships = np.empty((self.n_data, self.n_points))

        for i in range(self.n_data):
            d = self.comp_graphs[i]
            X_idxs = d.keys()
            # Resample points.
            X[i], X_memberships[i] = resample_points(
                    self.orig_points[X_idxs], self.areas[X_idxs])
            X_idxs_list.append(X_idxs)

        return X, X_idxs_list, self.graph_md5s, X_memberships


    def __iter__(self):
        self.index_in_epoch = 0
        self.perm = np.arange(self.n_data)
        np.random.shuffle(self.perm)
        return self


    def next(self):
        self.start = self.index_in_epoch * self.batch_size

        # FIXME:
        # Fix this when input placeholders have dynamic sizes.
        #self.end = min(self.start + self.batch_size, self.n_data)
        self.end = self.start + self.batch_size

        self.step_size = self.end - self.start
        self.index_in_epoch = self.index_in_epoch + 1

        # FIXME:
        # Fix this when input placeholders have dynamic sizes.
        #if self.start < self.n_data:
        if self.end <= self.n_data:
            shuffled_indices = self.perm[self.start:self.end]
            step_X, step_Y, _, _ =\
                    self.generate_random_samples(shuffled_indices)
            return step_X, step_Y
        else:
            raise StopIteration()

