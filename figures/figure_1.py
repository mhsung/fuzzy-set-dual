#!/usr/bin/env python
# Minhyuk Sung (mhsung@cs.stanford.edu)
# March 2018

import os, sys
BASE_DIR = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__))))

from collect_results import *


if __name__ == '__main__':
    synset_partial_idx_pairs = [
            ('Chair', 584)]

    top_k = 10

    out_name = os.path.splitext(os.path.basename(__file__))[0]
    out_dir = os.path.join(BASE_DIR, out_name)

    # NOTE:
    # Among the retrievals, 'ret_03', 'ret_04', 'ret_07', and 'ret_09' are
    # visualized in the paper.
    collect_complementarity_results(synset_partial_idx_pairs, top_k, out_dir)

