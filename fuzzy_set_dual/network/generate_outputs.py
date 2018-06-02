# Minhyuk Sung (mhsung@cs.stanford.edu)
# March 2018

import os, sys
BASE_DIR = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(BASE_DIR, '..', '..', 'network_utils'))

from dataset import Dataset
import math
import numpy as np
import tensorflow as tf
import tf_util


def inclusion_errors(X, Y):
    # X includes Y.
    # E = || relu(Y - X) ||^2
    errors = np.sum(np.square(np.maximum(Y - X, 0.0)), -1)
    return errors


def predict(X, sess, net, func):
    n_data = len(X)
    n_batches_in_epoch = int(math.ceil(float(n_data) / net.batch_size))
    X_net = None

    for index_in_epoch in range(n_batches_in_epoch):
        start = index_in_epoch * net.batch_size
        end = min(start + net.batch_size, n_data)
        n_step_size = end - start
        step_X = X[start:end]

        # NOTE:
        # Add dummy.
        if n_step_size < net.batch_size:
            assert(X.ndim > 1)
            dummy_shape = list(X.shape)
            dummy_shape[0] = net.batch_size - n_step_size
            step_X = np.vstack((step_X, np.zeros(dummy_shape)))

        if func == 'f':
            step_X_net = sess.run(net.fX, feed_dict={
                net.X: step_X, net.is_training: False})
        elif func == 'g':
            step_X_net = sess.run(net.gX, feed_dict={
                net.X: step_X, net.is_training: False})
        else:
            assert(False)

        # NOTE:
        # Remove dummy data.
        step_X_net = step_X_net[:n_step_size]

        if index_in_epoch == 0:
            X_net = step_X_net
        else:
            X_net = np.vstack((X_net, step_X_net))

    return X_net


def compute_component_embed_coords(sess, net, data):
    X, X_idxs = data.get_data_components()
    fX = predict(X, sess, net, 'f')
    gX = predict(X, sess, net, 'g')

    return X, X_idxs, fX, gX

