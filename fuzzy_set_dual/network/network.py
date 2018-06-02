# Minhyuk Sung (mhsung@cs.stanford.edu)
# March 2018

import os, sys
BASE_DIR = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__))))

from architectures import *
from dataset import Dataset
import math
import numpy as np
import tensorflow as tf


class Network(object):
    def __init__(self, arch, n_points, D, alpha, batch_size,
            optimizer, init_learning_rate, momentum, decay_step, decay_rate,
            relative):
        self.arch = arch
        self.n_points = n_points
        self.D = D
        self.alpha = alpha
        self.batch_size = batch_size
        self.init_learning_rate = init_learning_rate
        self.decay_step = decay_step
        self.decay_rate = decay_rate
        self.relative = relative

        self.graph = tf.Graph()
        with self.graph.as_default() as g:
            self.global_step = tf.Variable(0)

            self.bn_decay = get_batch_norm_decay(
                    self.global_step, self.batch_size, self.decay_step)
            tf.summary.scalar('bn_decay', self.bn_decay)

            self.is_training = tf.placeholder(tf.bool, shape=())

            # Build network.
            self.build_nets(self.is_training, self.bn_decay)

            self.learning_rate = get_learning_rate(
                    self.init_learning_rate, self.global_step, self.batch_size,
                    self.decay_step, self.decay_rate)
            tf.summary.scalar('learning_rate', self.learning_rate)

            if optimizer == 'momentum':
                self.train_op = tf.train.MomentumOptimizer(
                        self.learning_rate, momentum=momentum).minimize(
                                self.loss, global_step=self.global_step)
            elif optimizer == 'adam':
                self.train_op = tf.train.AdamOptimizer(
                        self.learning_rate).minimize(
                                self.loss, global_step=self.global_step)
            else:
                raise AssertionError

            # Define merged summary.
            self.summary = tf.summary.merge_all()

            # Define saver.
            self.saver = tf.train.Saver(max_to_keep=0)


    def compute_energy(self, X_net, Y_net):
        # X includes Y => X >= Y
        # E = || relu(Y - X) ||^2
        E_XY = tf.reduce_sum(tf.square(tf.nn.relu(Y_net - X_net)), -1)
        return E_XY


    def positive_negative_sample_losses(self, X_net, Y_net):
        # X includes Y.

        # (B, 1, D)
        X_net_exp = tf.expand_dims(X_net, 1)
        # (1, B, D)
        Y_net_exp = tf.expand_dims(Y_net, 0)
        # (B, B)
        E_XY = self.compute_energy(X_net_exp, Y_net_exp)
        S_XY = -E_XY

        diag_S_XY = tf.diag_part(S_XY)

        # Take off-diagonal pairs as negative samples.
        np_off_diag = np.ones((self.batch_size, self.batch_size),
                dtype=np.float32)
        for i in range(self.batch_size): np_off_diag[i,i] = 0.
        tf_off_diag = tf.constant(np_off_diag)
        n_off_diag = (self.batch_size) * (self.batch_size - 1)

        # In S, diagonal elements should be *larger* than any other element
        # on the same row and column (with the margin alpha).
        diag_S_XY_row_exp = tf.expand_dims(diag_S_XY, 1)
        diag_S_XY_col_exp = tf.expand_dims(diag_S_XY, 0)

        row_losses = tf.nn.relu((self.alpha + S_XY) - diag_S_XY_row_exp)
        col_losses = tf.nn.relu((self.alpha + S_XY) - diag_S_XY_col_exp)

        row_loss = tf.reduce_sum(row_losses * tf_off_diag) / n_off_diag
        col_loss = tf.reduce_sum(col_losses * tf_off_diag) / n_off_diag

        loss = 0.5 * (row_loss + col_loss)

        ranks = tf.count_nonzero(tf.greater_equal(
            S_XY, tf.expand_dims(diag_S_XY, 1)), 1)
        is_top_rank = tf.less_equal(ranks, 1)

        accu = tf.reduce_mean(tf.cast(is_top_rank, tf.float32))

        return loss, accu


    def positive_negative_sample_losses_with_tol(self, X_net, Y_net):
        # X includes Y.

        # (B, 1, D)
        X_net_exp = tf.expand_dims(X_net, 1)
        # (1, B, D)
        Y_net_exp = tf.expand_dims(Y_net, 0)
        # (B, B)
        E_XY = self.compute_energy(X_net_exp, Y_net_exp)

        # Take off-diagonal pairs as negative samples.
        np_off_diag = np.ones((self.batch_size, self.batch_size),
                dtype=np.float32)
        for i in range(self.batch_size): np_off_diag[i,i] = 0.
        tf_off_diag = tf.constant(np_off_diag)
        n_off_diag = (self.batch_size) * (self.batch_size - 1)

        # pos_E <= pos_tol => L = relu(pos_E - pos_tol)
        n_pos = float(self.batch_size)
        pos_tol = self.tol - 0.5 * self.alpha
        pos_loss = tf.reduce_sum(tf.nn.relu(tf.diag_part(E_XY) - pos_tol)) / \
                n_pos

        # neg_E >= neg_tol => L = relu(neg_tol - neg_E)
        n_neg = float(n_off_diag)
        neg_tol = self.tol + 0.5 * self.alpha
        neg_loss = tf.reduce_sum(tf.nn.relu(neg_tol - E_XY) * tf_off_diag) / \
                n_neg

        pos_corr = tf.less(tf.diag_part(E_XY), self.tol)
        pos_accu = tf.reduce_sum(tf.cast(pos_corr, tf.float32)) / n_pos

        neg_corr = tf.greater(E_XY, self.tol)
        neg_accu = tf.reduce_sum(tf.cast(neg_corr, tf.float32) * \
                tf_off_diag) / n_neg

        return pos_loss, pos_accu, neg_loss, neg_accu


    def build_nets(self, is_training, bn_decay):
        # FIXME:
        # Make the placeholders to have dynamic sizes.

        self.X = tf.placeholder(tf.float32,
                shape=[self.batch_size, self.n_points, 3])
        self.Y = tf.placeholder(tf.float32,
                shape=[self.batch_size, self.n_points, 3])

        scope = 'func_f'
        with tf.variable_scope(scope) as sc:
            self.fX = build_architecture(
                    self.arch, self.X, self.D, is_training, bn_decay, scope)

        with tf.variable_scope(scope, reuse=True) as sc:
            self.fY = build_architecture(
                    self.arch, self.Y, self.D, is_training, bn_decay, scope)

        scope = 'func_g'
        with tf.variable_scope(scope) as sc:
            self.gX = build_architecture(
                    self.arch, self.X, self.D, is_training, bn_decay, scope)

        with tf.variable_scope(scope, reuse=True) as sc:
            self.gY = build_architecture(
                    self.arch, self.Y, self.D, is_training, bn_decay, scope)


        self.tol = tf.Variable(self.alpha, name='tol')
        self.clip_tol_op = self.tol.assign(
            tf.clip_by_value(self.tol, self.alpha, 5. * self.alpha))


        self.E_XY = self.compute_energy(self.fX, self.gY)
        self.E_YX = self.compute_energy(self.fY, self.gX)


        if self.relative:
            self.XY_loss, self.XY_accu = \
                    self.positive_negative_sample_losses(self.fX, self.gY)
            self.YX_loss, self.YX_accu = \
                    self.positive_negative_sample_losses(self.fY, self.gX)
        else:
            XY_pos_loss, XY_pos_accu, XY_neg_loss, XY_neg_accu = \
                    self.positive_negative_sample_losses_with_tol(
                            self.fX, self.gY)
            YX_pos_loss, YX_pos_accu, YX_neg_loss, YX_neg_accu = \
                    self.positive_negative_sample_losses_with_tol(
                            self.fY, self.gX)

            self.XY_loss = 0.5 * (XY_pos_loss + XY_neg_loss)
            self.YX_loss = 0.5 * (YX_pos_loss + YX_neg_loss)
            self.XY_accu = 0.5 * (XY_pos_accu + XY_neg_accu)
            self.YX_accu = 0.5 * (YX_pos_accu + YX_neg_accu)

            self.pos_loss = 0.5 * (XY_pos_loss + YX_pos_loss)
            self.neg_loss = 0.5 * (XY_neg_loss + YX_neg_loss)
            self.pos_accu = 0.5 * (XY_pos_accu + YX_pos_accu)
            self.neg_accu = 0.5 * (XY_neg_accu + YX_neg_accu)

        with tf.name_scope('loss'):
            self.loss = 0.5 * (self.XY_loss + self.YX_loss) * 100.
        tf.summary.scalar('loss', self.loss)

        with tf.name_scope('accuracy'):
            self.accuracy = 0.5 * (self.XY_accu + self.YX_accu)
        tf.summary.scalar('accuracy', self.accuracy)

