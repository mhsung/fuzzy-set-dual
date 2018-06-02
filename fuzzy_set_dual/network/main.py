#!/usr/bin/env python
# Minhyuk Sung (mhsung@cs.stanford.edu)
# March 2018

import os, sys
BASE_DIR = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(BASE_DIR, '..', '..'))

from global_variables import *
from dataset import Dataset
from datetime import datetime
from evaluate import *
from network import Network
from train_util import *
import gflags
import math
import numpy as np
import random
import tensorflow as tf


FLAGS = gflags.FLAGS
gflags.DEFINE_string('in_model_dirs', '', '')
gflags.DEFINE_string('in_model_scopes', '', '')
gflags.DEFINE_string('out_model_dir', 'model', '')
gflags.DEFINE_string('out_dir', 'outputs', '')
gflags.DEFINE_string('log_dir', 'log', '')

gflags.DEFINE_string('arch', 'vanilla', 'vanilla or pp [default: pp]')

gflags.DEFINE_bool('train', False, '')
gflags.DEFINE_string('optimizer', 'adam',\
        'adam or momentum [default: adam]')
gflags.DEFINE_float('init_learning_rate', 0.001,\
        'Initial learning rate [default: 0.001]')
gflags.DEFINE_float('momentum', 0.9,\
        'Initial learning rate [default: 0.9]')
gflags.DEFINE_float('decay_step', 200000,\
        'Decay step for lr decay [default: 50000]')
gflags.DEFINE_float('decay_rate', 0.7,\
        'Decay rate for lr decay [default: 0.8]')

gflags.DEFINE_integer('dim', 100, '')
gflags.DEFINE_float('alpha', 0.05, '')
gflags.DEFINE_bool('centerize', False, 'Centerize partial objects.')
gflags.DEFINE_bool('relative', False, '')

gflags.DEFINE_integer('n_epochs', 1000, '')
gflags.DEFINE_integer('batch_size', 32, '')
gflags.DEFINE_integer('snapshot_epoch', 100, 'Use pairwise ranking loss.')


def print_params(n_points):
    print('==== PARAMS ====')
    print(' - Architecture: {}'.format(FLAGS.arch))
    print(' - # epochs: {:d}'.format(FLAGS.n_epochs))
    print(' - batch size: {:d}'.format(FLAGS.batch_size))
    print(' - D: {:d}'.format(FLAGS.dim))
    print(' - alpha: {:f}'.format(FLAGS.alpha))
    print(' - centerize: {}'.format(FLAGS.centerize))
    print(' - relative: {}'.format(FLAGS.relative))


def load_model(sess, in_model_dir, exclude=''):
    # Read variables names in checkpoint.
    var_names = [x for x,_ in tf.contrib.framework.list_variables(in_model_dir)]

    # Find variables with given names.
    # HACK:
    # Convert unicode to string and remove postfix ':0'.
    var_list = [x for x in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)\
            if str(x.name)[:-2] in var_names]

    if exclude != '':
        var_list = [x for x in var_list if exclude not in x.name]
    #print([x.name for x in var_list])

    saver = tf.train.Saver(var_list)

    ckpt = tf.train.get_checkpoint_state(in_model_dir)
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
        print ("Loaded '{}'.".format(ckpt.model_checkpoint_path))
    else:
        print ("Failed to loaded '{}'.".format(in_model_dir))
        return False
    return True


if __name__ == '__main__':
    FLAGS(sys.argv)
    tf.set_random_seed(0)
    np.random.seed(0)
    random.seed(0)

    # NOTE:
    # Read all point cloud data and select a subset using a train/test 
    # connected component pair list file.

    train_data = Dataset(
            g_component_train_pairs_file,
            g_component_all_component_labels_file,
            g_component_all_centered_point_clouds_file,
            g_component_all_positions_file,
            g_component_all_areas_file,
            FLAGS.batch_size,
            FLAGS.centerize)

    test_data = Dataset(
            g_component_test_pairs_file,
            g_component_all_component_labels_file,
            g_component_all_centered_point_clouds_file,
            g_component_all_positions_file,
            g_component_all_areas_file,
            FLAGS.batch_size,
            FLAGS.centerize)

    print_params(train_data.n_points)

    net = Network(FLAGS.arch, train_data.n_points, FLAGS.dim, FLAGS.alpha,
            FLAGS.batch_size, FLAGS.optimizer, FLAGS.init_learning_rate,
            FLAGS.momentum, FLAGS.decay_step, FLAGS.decay_rate, FLAGS.relative)


    config = tf.ConfigProto()
    config.allow_soft_placement = True
    config.gpu_options.allow_growth = True

    with tf.Session(config=config, graph=net.graph) as sess:
        sess.run(tf.global_variables_initializer(), {net.is_training: True})

        if FLAGS.in_model_dirs:
            exclude = ''
            for in_model_dir in FLAGS.in_model_dirs.split(','):
                assert(load_model(sess, in_model_dir, exclude))

        if FLAGS.train:
            train(sess, net, train_data, test_data, n_epochs=FLAGS.n_epochs,
                    snapshot_epoch=FLAGS.snapshot_epoch,
                    model_dir=FLAGS.out_model_dir, log_dir=FLAGS.log_dir,
                    data_name=g_synset, output_generator=None)

        tol = sess.run(net.tol, feed_dict={net.is_training: False})
        print(' - tol: {:5f}'.format(tol))

        train_loss, train_accuracy, _ = evaluate(sess, net, train_data)
        test_loss, test_accuracy, _ = evaluate(sess, net, test_data)
        msg = "|| Train Loss: {:6f}".format(train_loss)
        msg += " | Train Accu: {:5f}".format(train_accuracy)
        msg += " | Test Loss: {:6f}".format(test_loss)
        msg += " | Test Accu: {:5f}".format(test_accuracy)
        msg += " ||"
        print(msg)

        if FLAGS.train:
            # Save training result.
            if not os.path.exists(FLAGS.out_dir): os.makedirs(FLAGS.out_dir)
            out_file = os.path.join(FLAGS.out_dir, '{}.txt'.format(
                datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
            with open(out_file, 'w') as f:
                f.write(msg + '\n')
            print("Saved '{}'.".format(out_file))
        else:
            if FLAGS.relative:
                retrieve_and_evaluate_complements(sess, net, test_data,
                        FLAGS.centerize, FLAGS.out_dir)
            else:
                retrieve_interchangeables(sess, net, test_data,
                        FLAGS.centerize, FLAGS.out_dir)
                compute_part_interchangeability(sess, net, test_data,
                        FLAGS.centerize, FLAGS.out_dir)

