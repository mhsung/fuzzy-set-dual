#!/usr/bin/env python
# Minhyuk Sung (mhsung@cs.stanford.edu)
# March 2018

import os, sys
BASE_DIR = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__))))

from collect_results import *


if __name__ == '__main__':
    synset_partial_idx_pairs = [
            ('Airplane', 579),
            ('Car', 57),
            ('Chair', 7251),
            ('Guitar', 5),
            ('Lamp', 117),
            ('Rifle', 371),
            ('Sofa', 1184),
            ('Table', 1414),
            ('Watercraft', 294)]

    top_k = 5

    out_name = os.path.splitext(os.path.basename(__file__))[0]
    out_dir = os.path.join(BASE_DIR, out_name)

    collect_complementarity_results(synset_partial_idx_pairs, top_k, out_dir)

