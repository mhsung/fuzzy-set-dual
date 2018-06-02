#!/usr/bin/env python
# Minhyuk Sung (mhsung@cs.stanford.edu)
# March 2017

import os, sys
BASE_DIR = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__))))

# NOTE:
# Install matplotlib:
# python install matplotlib
from matplotlib import pyplot as plt

from scipy.stats import rankdata
import glob
import numpy as np


def plot_part_neighbors(synset, out_dir):
    if not os.path.exists(out_dir): os.makedirs(out_dir)

    exp_root_dir = os.path.abspath(os.path.join(BASE_DIR, '..',
        'fuzzy_set_dual', 'experiments'))
    assert(os.path.exists(exp_root_dir))

    labels_file = os.path.join(exp_root_dir, synset, 'vanilla_100_centerize',
            'outputs', 'part_labels.txt')
    assert(os.path.exists(labels_file))
    labels = np.genfromtxt(labels_file, dtype=int)
    unique_labels = np.unique(labels)
    n_labels = unique_labels.size
    print('# labels: {:d}'.format(n_labels))

    dists_file = os.path.join(exp_root_dir, synset, 'vanilla_100_centerize',
            'outputs', 'part_interchangeability_distances.npy')
    print(dists_file)
    assert(os.path.exists(dists_file))
    dists = np.load(dists_file)

    n_all_parts = dists.shape[0]
    print('# all parts: {:d}'.format(n_all_parts))

    props = np.linspace(0.01, 0.1, 10)
    n_props = len(props)
    print(props)

    purities = np.zeros((n_all_parts, n_props))
    for i in range(n_all_parts):
        ranks = rankdata(dists[i], method='min')

        for k in range(n_props):
            prop = int(props[k] * n_all_parts)
            idxs = np.where(ranks <= prop)[0]
            purities[i,k] = \
                    float(np.sum(labels[idxs] == labels[i])) / idxs.size

    plt.clf()
    plt.figure(figsize=(6, 6), dpi=100)
    plt.plot(props, np.mean(purities, axis=0), label='Ours', color='g')

    plt.title(synset, fontsize=28)
    plt.xlabel('Nearest neighbor percentage', fontsize=20)
    plt.ylabel('Ratio of correct semantic labels', fontsize=20)
    plt.xticks(props)
    plt.yticks([0.7, 1.])

    if synset == 'Lamp':
        plt.legend(loc=2, fontsize=20)
    else:
        plt.legend(loc=3, fontsize=20)

    out_file = os.path.join(out_dir, '{}.png'.format(synset))
    plt.savefig(out_file)
    print("Saved '{}'.".format(out_file))
    return out_file


if __name__ == '__main__':
    all_synsets = ['Airplane', 'Car', 'Chair', 'Guitar', 'Lamp', 'Table']

    out_name = os.path.splitext(os.path.basename(__file__))[0]
    out_dir = os.path.join(BASE_DIR, out_name)

    for synset in all_synsets:
        plot_part_neighbors(synset, out_dir)

