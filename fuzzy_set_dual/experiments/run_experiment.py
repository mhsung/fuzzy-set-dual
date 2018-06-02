#!/usr/bin/env python
# Minhyuk Sung (mhsung@cs.stanford.edu)
# March 2018

import os, sys
BASE_DIR = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.join(BASE_DIR, '..', '..'))

from global_variables import *
import gflags

FLAGS = gflags.FLAGS

gflags.DEFINE_string('arch', 'vanilla', 'vanilla or pp [default: vanilla]')

gflags.DEFINE_bool('train', False, '')
gflags.DEFINE_integer('dim', 100, '')
gflags.DEFINE_bool('centerize', True, 'Centerize partial objects.')
gflags.DEFINE_bool('relative', False, 'Use pairwise ranking loss.')


if __name__ == '__main__':
    FLAGS(sys.argv)

    root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
            g_synset, '{}_{:d}'.format(FLAGS.arch, FLAGS.dim))
    if FLAGS.centerize: root_dir += '_centerize'
    if FLAGS.relative: root_dir += '_relative'
    if not os.path.exists(root_dir): os.makedirs(root_dir)

    model_dir = os.path.join(root_dir, 'model')
    out_dir = os.path.join(root_dir, 'outputs')
    log_dir = os.path.join(root_dir, 'log')


    cmd = os.path.join(os.path.dirname(os.path.abspath(__file__)),
            '..', 'network', 'main.py') + ' \\\n\t'
    cmd += '-arch=' + str(FLAGS.arch) + ' \\\n\t'
    cmd += '-dim=' + str(FLAGS.dim) + ' \\\n\t'
    cmd += '-centerize=' + str(FLAGS.centerize) + ' \\\n\t'
    cmd += '-relative=' + str(FLAGS.relative) + ' \\\n\t'

    if FLAGS.train:
        cmd += '-train \\\n\t'
        cmd += '-out_model_dir=' + model_dir + ' \\\n\t'
    else:
        cmd += '-in_model_dirs=' + model_dir + ' \\\n\t'
    cmd += '-out_dir=' + out_dir + ' \\\n\t'
    cmd += '-log_dir=' + log_dir

    print(cmd)
    os.system(cmd)

