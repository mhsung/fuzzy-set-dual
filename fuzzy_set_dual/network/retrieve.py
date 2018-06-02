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
from render_mesh import *
from resample_points import resample_points
import itertools
import numpy as np
import random


def get_all_component_graph_idxs(data):
    comp_graph_idxs = np.full(data.n_components, -1, dtype=int)

    for graph_idx in range(data.n_data):
        G = data.comp_graphs[graph_idx]
        for i in list(G.nodes):
            comp_graph_idxs[i] = graph_idx

    return comp_graph_idxs


def get_all_partial_shapes(data):
    n_all_subsets = np.sum([len(x) for x in data.comp_all_subsets])

    X = np.empty((n_all_subsets, data.n_points, 3))
    X_graph_idxs = [None] * n_all_subsets
    X_comp_idxs_list = [None] * n_all_subsets

    count = 0
    for graph_idx in range(data.n_data):
        for X_comp_idxs in data.comp_all_subsets[graph_idx]:
            X[count] = resample_points(data.orig_points[X_comp_idxs],
                    data.areas[X_comp_idxs])
            X_graph_idxs[count] = graph_idx
            X_comp_idxs_list[count] = X_comp_idxs
            count += 1

    return X, X_graph_idxs, X_comp_idxs_list


def find_included_subgraphs(sess, net, data, X_graph_idx, Y_graph_idxs,
        Y_comp_idxs_list, E):
    n_subsets = len(Y_graph_idxs)
    assert(len(Y_comp_idxs_list) == n_subsets)
    assert(E.size == n_subsets)
    print('# tested subsets: {:d}'.format(n_subsets))


    Y_graph_subset_idxs = {}
    for i in range(n_subsets):
        graph_idx = Y_graph_idxs[i]
        if graph_idx not in Y_graph_subset_idxs:
            Y_graph_subset_idxs[graph_idx] = []
        Y_graph_subset_idxs[graph_idx].append(i)


    # Find the minimial size minimum error complement in the same object.
    retrievals = []

    for graph_idx, subset_idxs in Y_graph_subset_idxs.iteritems():
        retrieval = {}
        retrieval['graph_idx'] = graph_idx

        '''
        # Find the minimal size of subsets.
        min_size = np.amin([len(Y_comp_idxs_list[i]) for i in subset_idxs])
        min_subset_idxs = [i for i in subset_idxs \
                if len(Y_comp_idxs_list[i]) == min_size]
        '''

        # Find the minimum error subset of them.
        #min_subset_idx = min_subset_idxs[np.argmin(E[min_subset_idxs])]
        min_subset_idx = subset_idxs[np.argmin(E[subset_idxs])]
        min_subset = Y_comp_idxs_list[min_subset_idx]

        retrieval['minset'] = min_subset
        retrieval['minset_E'] = E[min_subset_idx]

        # Find all optionals of minimum set.
        retrieval['optionals'] = []
        retrieval['optionals_E'] = []
        for i in subset_idxs:
            if set(min_subset) < set(Y_comp_idxs_list[i]):
                set_diff = list(set(Y_comp_idxs_list[i]) - set(min_subset))
                retrieval['optionals'].append(Y_comp_idxs_list[i])
                retrieval['optionals_E'].append(E[i])

        retrievals.append(retrieval)

    print('# final subsets: {:d}'.format(len(retrievals)))

    # Sort.
    idxs = np.argsort([x['minset_E'] for x in retrievals])
    retrievals = [retrievals[i] for i in idxs]

    return retrievals


def render_retrieved_subsets(data, query_idxs, retrievals, out_dir):
    kNumRetrievals = 31

    cmd = 'montage '

    out_file = os.path.join(out_dir, 'query.png')
    render_selected_components(data, query_idxs, out_file)
    os.system('convert {0} -frame 8x8 {0}'.format(out_file))
    cmd += (out_file + ' ')

    count = 0
    for retrieval in retrievals[:kNumRetrievals]:
        out_file = os.path.join(out_dir, '{:02d}.png'.format(count))

        ret_idxs = []
        ret_labels = []
        label = 1

        for i in retrieval['minset']:
            ret_idxs.append(i)
            ret_labels.append(label)

        '''
        for subset in retrieval['optionals']:
            label += 1
            for i in subset:
                ret_idxs.append(i)
                ret_labels.append(label)
        '''

        render_selected_components(data, ret_idxs, out_file, ret_labels)
        cmd += (out_file + ' ')
        count += 1

    out_file = out_dir + '.png'
    cmd += out_file
    os.system(cmd)
    assert(os.path.exists(out_file))
    print("Saved '{}'.".format(out_file))


def retrieve_partial_objects(sess, net, data, alpha, tol, out_dir):
    Y, Y_graph_idxs, Y_comp_idxs_list = get_all_partial_shapes(data)
    n_all_subsets = Y.shape[0]
    print('# all subsets: {:d}'.format(n_all_subsets))

    print('Compute embedding coordinates...')
    fY = predict(Y, sess, net, 'f')
    gY = predict(Y, sess, net, 'g')
    print('Done.')

    # Sample partial shapes.
    kNumSamples = 100
    perm = np.arange(n_all_subsets)
    np.random.shuffle(perm)

    X_idxs = perm[:kNumSamples]
    X = Y[X_idxs]
    fX = fY[X_idxs]
    gX = gY[X_idxs]
    X_graph_idxs = [Y_graph_idxs[i] for i in X_idxs]
    X_comp_idxs_list = [Y_comp_idxs_list[i] for i in X_idxs]
    print('# X subsets: {:d}'.format(X.shape[0]))

    query_subsets_file = os.path.join(out_dir, 'query_subsets.txt')
    with open(query_subsets_file, 'w') as f:
        for X_comp_idxs in X_comp_idxs_list:
            f.write(' '.join([str(x) for x in X_comp_idxs]) + '\n')
    print("Saved '{}'.".format(query_subsets_file))

    print('Compute inclusion errors...')
    E_XY = inclusion_errors(np.expand_dims(fX, 1), np.expand_dims(gY, 0))
    E_YX = inclusion_errors(np.expand_dims(fY, 1), np.expand_dims(gX, 0))
    print('Done.')


    for i in range(kNumSamples):
        i_E_XY = E_XY[i, :]
        i_E_YX = E_YX[:, i]
        idxs = np.where(np.logical_and(i_E_XY < tol, i_E_YX < tol))[0]

        # NOTE:
        # Skip subsets in the query object.
        idxs = np.array([j for j in idxs if Y_graph_idxs[j] != X_graph_idxs[i]])

        i_Y_graph_idxs = [Y_graph_idxs[j] for j in idxs]
        i_Y_comp_idxs_list = [Y_comp_idxs_list[j] for j in idxs]
        i_E = 0.5 * (i_E_XY[idxs] + i_E_YX[idxs])

        retrievals = find_included_subgraphs(sess, net, data, X_graph_idxs[i],
                i_Y_graph_idxs, i_Y_comp_idxs_list, i_E)

        i_out_dir = os.path.join(out_dir, 'retrievals', '{:04d}'.format(i))
        if not os.path.exists(i_out_dir): os.makedirs(i_out_dir)
        render_retrieved_subsets(data, X_comp_idxs_list[i], retrievals,
                i_out_dir)


def render_retrieved_components(data, query_idx, retrieved_idxs, out_dir):
    kNumRetrievals = 21
    kInterval = 5

    cmd = 'montage '

    out_file = os.path.join(out_dir, 'query.png')
    render_selected_components(data, [query_idx], out_file)
    os.system('convert {0} -frame 8x8 {0}'.format(out_file))
    cmd += (out_file + ' ')

    for i in range(kNumRetrievals):
        retrieved_idx = retrieved_idxs[i*kInterval]
        out_file = os.path.join(out_dir, '{:02d}.png'.format(i))
        render_selected_components(data, [retrieved_idx], out_file, 1)
        cmd += (out_file + ' ')

    out_file = out_dir + '.png'
    cmd += out_file
    os.system(cmd)
    assert(os.path.exists(out_file))
    print("Saved '{}'.".format(out_file))


def retrieve_replaceable_components(sess, net, data, alpha, tol, centerize,
        out_dir):
    if centerize:
        all_point_clouds = data.centered_points
    else:
        all_point_clouds = data.orig_points

    n_all_components = all_point_clouds.shape[0]
    print('# all components: {:d}'.format(n_all_components))

    print('Compute embedding coordinates...')
    g_coords = predict(all_point_clouds, sess, net, 'g')
    print('Done.')

    # Sample components.
    kNumSamples = 100
    perm = np.arange(n_all_components)
    np.random.shuffle(perm)

    count = 0
    for query_idx in perm[:kNumSamples]:
        norms = np.linalg.norm(np.maximum(
            np.expand_dims(g_coords[query_idx], 0), g_coords), axis=-1)
        assert(norms.size == n_all_components)

        ret_idxs = np.argsort(norms)
        ret_idxs = [x for x in ret_idxs if x != query_idx]

        ret_our_dir = os.path.join(out_dir, 'replaceability', '{:04d}'.format(
            count))
        if not os.path.exists(ret_our_dir): os.makedirs(ret_our_dir)
        render_retrieved_components(data, query_idx, ret_idxs, ret_our_dir)

        count += 1

