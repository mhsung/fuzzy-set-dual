#!/usr/bin/env python
# Minhyuk Sung (mhsung@cs.stanford.edu)
# March 2018

import os, sys
BASE_DIR = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__))))

from collect_results import *


if __name__ == '__main__':
    # NOTE:
    # The comment of each row indicates the retrieval outputs visualized in the
    # paper.
    synset_partial_idx_pairs = [
            ('Chair', 6206),    # ret_00, ret_03, ret_05
            ('Table', 2470),    # ret_01, ret_06, ret_08
            ('Table', 2662),    # ret_00, ret_02, ret_07
            ('Airplane', 1087)  # ret_00, ret_03, ret_06
            ]

    top_k = 10

    out_name = os.path.splitext(os.path.basename(__file__))[0]
    out_dir = os.path.join(BASE_DIR, out_name)

    collect_complementarity_results(synset_partial_idx_pairs, top_k, out_dir)

