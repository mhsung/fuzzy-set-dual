#!/usr/bin/env python
# Minhyuk Sung (mhsung@cs.stanford.edu)
# March 2018

import os, sys
BASE_DIR = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__))))

from collect_results import *


if __name__ == '__main__':
    synset_partial_idx_pairs = [
            ('Airplane', 2550),
            ('Car', 1869),
            ('Chair', 1189),
            ('Guitar', 1),
            ('Lamp', 1738),
            ('Rifle', 565),
            ('Sofa', 2867),
            ('Table', 2940),
            ('Watercraft', 908)]

    top_k = 5

    out_name = os.path.splitext(os.path.basename(__file__))[0]
    out_dir = os.path.join(BASE_DIR, out_name)

    collect_interchangeability_results(synset_partial_idx_pairs, top_k, out_dir)

