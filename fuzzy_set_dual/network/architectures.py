# Minhyuk Sung (mhsung@cs.stanford.edu)
# March 2018

import os, sys
BASE_DIR = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(BASE_DIR, '..', '..', 'network_utils'))
sys.path.append(os.path.join(BASE_DIR, '..', '..', 'network_utils', 'pointnet2'))
sys.path.append(os.path.join(BASE_DIR, '..', '..', 'network_utils', 'tf_ops'))

from pointnet_util import pointnet_sa_module, pointnet_fp_module
import tensorflow as tf
import tf_util


def get_batch_norm_decay(global_step, batch_size, bn_decay_step):
    BN_INIT_DECAY = 0.5
    BN_DECAY_RATE = 0.5
    BN_DECAY_CLIP = 0.99

    bn_momentum = tf.train.exponential_decay(
                    BN_INIT_DECAY,
                    global_step*batch_size,
                    bn_decay_step,
                    BN_DECAY_RATE,
                    staircase=True)

    bn_decay = tf.minimum(BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay


def get_learning_rate(init_learning_rate, global_step, batch_size,
        decay_step, decay_rate):
    learning_rate = tf.train.exponential_decay(
                        init_learning_rate,
                        global_step*batch_size,
                        decay_step,
                        decay_rate,
                        staircase=True)
    learing_rate = tf.maximum(learning_rate, 0.00001)
    return learning_rate


def build_architecture(arch, X, out_dim, is_training, bn_decay, scope):
    if arch == 'vanilla':
        net = build_pointnet(X, out_dim, is_training, bn_decay, scope)
    elif arch == 'pp':
        #net = build_pointnet_pp_ssg(X, out_dim, is_training, bn_decay, scope)
        net = build_pointnet_pp_msg(X, out_dim, is_training, bn_decay, scope)
    else:
        print('[Error] Unknown architecture: {}'.format(arch))
        sys.exit(-1)

    # Abs.
    net = tf.abs(net)

    # Normalize.
    net = tf.nn.l2_normalize(net, -1)

    return net


def build_pointnet(X, out_dim, is_training, bn_decay, scope):
    n_points = X.get_shape()[1].value

    X_expanded = tf.expand_dims(X, -1)

    net = tf_util.conv2d(X_expanded, 64, [1,3], padding='VALID',
            stride=[1,1], bn=True, is_training=is_training,
            bn_decay=bn_decay, scope=scope+'_conv1')

    net = tf_util.conv2d(net, 64, [1,1], padding='VALID',
            stride=[1,1], bn=True, is_training=is_training,
            bn_decay=bn_decay, scope=scope+'_conv2')

    net = tf_util.conv2d(net, 64, [1,1], padding='VALID',
            stride=[1,1], bn=True, is_training=is_training,
            bn_decay=bn_decay, scope=scope+'_conv3')

    net = tf_util.conv2d(net, 128, [1,1], padding='VALID',
            stride=[1,1], bn=True, is_training=is_training,
            bn_decay=bn_decay, scope=scope+'_conv4')

    net = tf_util.conv2d(net, 1024, [1,1], padding='VALID',
            stride=[1,1], bn=True, is_training=is_training,
            bn_decay=bn_decay, scope=scope+'_conv5')

    net = tf_util.max_pool2d(net, [n_points, 1], padding='VALID',
            scope=scope+'_maxpool')

    net = tf.squeeze(net)

    net = tf_util.fully_connected(net, 512, bn=True,
            is_training=is_training, bn_decay=bn_decay,
            scope=scope+'_fc1')

    net = tf_util.fully_connected(net, 256, bn=True,
            is_training=is_training, bn_decay=bn_decay,
            scope=scope+'_fc2')

    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
            scope=scope+'_dp1')

    # Outputs.
    net = tf_util.fully_connected(net, out_dim,
            activation_fn=None, scope=scope+'_out')

    return net


def build_pointnet_pp_ssg(X, out_dim, is_training, bn_decay, scope):
    batch_size = X.get_shape()[0].value

    l0_xyz = X
    l0_points = None

    # Set abstraction layers
    # Note: When using NCHW for layer 2, we see increased GPU memory usage (in TF1.4).
    # So we only use NCHW for layer 1 until this issue can be resolved.
    l1_xyz, l1_points, l1_indices = pointnet_sa_module(l0_xyz, l0_points,
            npoint=512, radius=0.2, nsample=32, mlp=[64,64,128],
            mlp2=None, group_all=False, is_training=is_training,
            bn_decay=bn_decay, scope='layer1', use_nchw=True)

    l2_xyz, l2_points, l2_indices = pointnet_sa_module(l1_xyz, l1_points,
            npoint=128, radius=0.4, nsample=64, mlp=[128,128,256],
            mlp2=None, group_all=False, is_training=is_training,
            bn_decay=bn_decay, scope='layer2')

    l3_xyz, l3_points, l3_indices = pointnet_sa_module(l2_xyz, l2_points,
            npoint=None, radius=None, nsample=None, mlp=[256,512,1024],
            mlp2=None, group_all=True, is_training=is_training,
            bn_decay=bn_decay, scope='layer3')

    # Fully connected layers
    net = tf.reshape(l3_points, [batch_size, -1])

    net = tf_util.fully_connected(net, 512, bn=True,
            is_training=is_training, scope='fc1', bn_decay=bn_decay)

    net = tf_util.dropout(net, keep_prob=0.4,
            is_training=is_training, scope='dp1')

    net = tf_util.fully_connected(net, 256, bn=True,
            is_training=is_training, scope='fc2', bn_decay=bn_decay)

    net = tf_util.dropout(net, keep_prob=0.4,
            is_training=is_training, scope='dp2')

    net = tf_util.fully_connected(net, out_dim, activation_fn=None,
            scope='fc3')

    return net


def build_pointnet_pp_msg(X, out_dim, is_training, bn_decay, scope):
    batch_size = X.get_shape()[0].value

    l0_xyz = X
    l0_points = None

    # Set abstraction layers
    l1_xyz, l1_points = pointnet_sa_module_msg(l0_xyz, l0_points, 512,
            [0.1,0.2,0.4], [16,32,128],
            [[32,32,64], [64,64,128], [64,96,128]],
            is_training, bn_decay, scope='layer1', use_nchw=True)

    l2_xyz, l2_points = pointnet_sa_module_msg(l1_xyz, l1_points, 128,
            [0.2,0.4,0.8], [32,64,128],
            [[64,64,128], [128,128,256], [128,128,256]],
            is_training, bn_decay, scope='layer2')

    l3_xyz, l3_points, _ = pointnet_sa_module(l2_xyz, l2_points,
            npoint=None, radius=None, nsample=None,
            mlp=[256,512,1024], mlp2=None, group_all=True,
            is_training=is_training, bn_decay=bn_decay, scope='layer3')

    # Fully connected layers
    net = tf.reshape(l3_points, [batch_size, -1])

    net = tf_util.fully_connected(net, 512, bn=True,
            is_training=is_training, scope='fc1', bn_decay=bn_decay)

    net = tf_util.dropout(net, keep_prob=0.4,
            is_training=is_training, scope='dp1')

    net = tf_util.fully_connected(net, 256, bn=True,
            is_training=is_training, scope='fc2', bn_decay=bn_decay)

    net = tf_util.dropout(net, keep_prob=0.4,
            is_training=is_training, scope='dp2')

    net = tf_util.fully_connected(net, out_dim, activation_fn=None,
            scope='fc3')

    return net

