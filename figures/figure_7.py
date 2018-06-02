#!/usr/bin/env python
# Minhyuk Sung (mhsung@cs.stanford.edu)
# March 2018

import os, sys
BASE_DIR = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__))))

from collect_results import *


if __name__ == '__main__':
    synset_partial_idx_pairs = [
            ('Airplane', 1291),
            ('Car', 46),
            ('Chair', 4588),
            ('Guitar', 42),
            ('Lamp', 362),
            ('Rifle', 26),
            ('Sofa', 0),
            ('Table', 330),
            ('Watercraft', 1013)]

    top_k = 1

    out_name = os.path.splitext(os.path.basename(__file__))[0]
    out_dir = os.path.join(BASE_DIR, out_name)

    collect_complementarity_results(synset_partial_idx_pairs, top_k, out_dir)

