# Minhyuk Sung (mhsung@cs.stanford.edu)
# March 2018

import os, sys
BASE_DIR = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(BASE_DIR, '..'))

# NOTE: Do not use synset here.
os.environ['synset'] = ''
from global_variables import *
BASE_DIR = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__))))

from mesh_utils import *
import numpy as np
import pandas as pd


def get_partial_object_component_idxs(synset, partial_idx):
    # Read partial object component list.
    partial_test_component_idxs_file = os.path.join(g_data_root_dir,
            'partial_objects', synset, 'partial_test_component_idxs.txt')
    assert(os.path.exists(partial_test_component_idxs_file))

    partial_test_component_idxs = []
    with open(partial_test_component_idxs_file, 'r') as f:
        lines = f.read().splitlines()
        for line in lines:
            partial_test_component_idxs.append(
                    [int(x) for x in line.split(' ')])
    assert(partial_idx < len(partial_test_component_idxs))

    comp_idxs = partial_test_component_idxs[partial_idx]
    return comp_idxs


def merge_component_meshes(synset, comp_idxs, out_file):
    component_meshes_dir = os.path.join(
            g_data_root_dir, 'component_meshes', synset)
    assert(os.path.exists(component_meshes_dir))

    all_component_labels_file = os.path.join(g_data_root_dir, 'components',
            synset, 'component_all_component_labels.txt')
    assert(os.path.exists(all_component_labels_file))

    df = pd.read_csv(all_component_labels_file, index_col=False)
    all_md5s = df['md5'].tolist()
    all_comp_labels = [int(x) for x in df['idx'].tolist()]

    mesh_files = []
    for comp_idx in comp_idxs:
        md5 = all_md5s[comp_idx]
        comp_label = all_comp_labels[comp_idx]
        print(md5, comp_label)
        mesh_file = os.path.join(component_meshes_dir, md5,
                '{}.obj'.format(comp_label))
        assert(os.path.exists(mesh_file))
        mesh_files.append(mesh_file)

    merge_obj_files(mesh_files, out_file)
    print("Saved '{}'.".format(out_file))


def generate_partial_object_mesh(synset, partial_idx, out_file):
    comp_idxs = get_partial_object_component_idxs(synset, partial_idx)
    merge_component_meshes(synset, comp_idxs, out_file)


def collect_complementarity_results(
        synset_partial_idx_pairs, top_k, out_dir):
    if not os.path.exists(out_dir): os.makedirs(out_dir)

    exp_root_dir = os.path.abspath(os.path.join(BASE_DIR, '..',
        'fuzzy_set_dual', 'experiments'))
    assert(os.path.exists(exp_root_dir))

    for i, (synset, query_idx) in enumerate(synset_partial_idx_pairs):
        i_out_dir = os.path.join(out_dir, '{:02d}'.format(i))
        if not os.path.exists(i_out_dir): os.makedirs(i_out_dir)

        # Save query partial object.
        query_file = os.path.join(i_out_dir, 'query.obj')
        generate_partial_object_mesh(synset, query_idx, query_file)

        # Read retrieval results.
        retrieval_file = os.path.join(exp_root_dir, synset,
                'vanilla_100_centerize_relative', 'outputs',
                'complementarity_retrievals.txt')
        assert(os.path.exists(retrieval_file))
        retrievals = np.genfromtxt(retrieval_file, dtype=int)

        for k in range(top_k):
            ret_idx = retrievals[query_idx][k]

            # Save ret partial object.
            ret_file = os.path.join(i_out_dir, 'ret_{:02d}.obj'.format(k))
            generate_partial_object_mesh(synset, ret_idx, ret_file)


def collect_interchangeability_results(
        synset_partial_idx_pairs, top_k, out_dir):
    if not os.path.exists(out_dir): os.makedirs(out_dir)

    exp_root_dir = os.path.abspath(os.path.join(BASE_DIR, '..',
        'fuzzy_set_dual', 'experiments'))
    assert(os.path.exists(exp_root_dir))

    for i, (synset, query_idx) in enumerate(synset_partial_idx_pairs):
        i_out_dir = os.path.join(out_dir, '{:02d}'.format(i))
        if not os.path.exists(i_out_dir): os.makedirs(i_out_dir)

        # Save query partial object.
        query_file = os.path.join(i_out_dir, 'query.obj')
        generate_partial_object_mesh(synset, query_idx, query_file)

        # Read retrieval results.
        retrieval_file = os.path.join(exp_root_dir, synset,
                'vanilla_100_centerize', 'outputs',
                'interchangeability_retrievals.txt')
        assert(os.path.exists(retrieval_file))
        retrievals = np.genfromtxt(retrieval_file, dtype=int)

        for k in range(top_k):
            ret_idx = retrievals[query_idx][k]

            # Save ret partial object.
            ret_file = os.path.join(i_out_dir, 'ret_{:02d}.obj'.format(k))
            generate_partial_object_mesh(synset, ret_idx, ret_file)

