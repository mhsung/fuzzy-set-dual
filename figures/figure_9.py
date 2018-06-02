#!/usr/bin/env python
# Minhyuk Sung (mhsung@cs.stanford.edu)
# March 2018

import os, sys
BASE_DIR = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__))))

from collect_results import *


if __name__ == '__main__':
    synset_partial_idx_pairs = [
            ('Airplane', 911),
            ('Car', 2147),
            ('Chair', 5323),
            ('Lamp', 1724),
            ('Rifle', 585),
            ('Sofa', 2459)]

    top_k = 1

    out_name = os.path.splitext(os.path.basename(__file__))[0]
    out_dir = os.path.join(BASE_DIR, out_name)

    collect_interchangeability_results(synset_partial_idx_pairs, top_k, out_dir)

