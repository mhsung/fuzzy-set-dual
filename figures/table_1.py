#!/usr/bin/env python
# Minhyuk Sung (mhsung@cs.stanford.edu)
# March 2017

import os, sys
BASE_DIR = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__))))


import numpy as np


def get_complementarity_stats(synset):
    exp_root_dir = os.path.abspath(os.path.join(BASE_DIR, '..',
        'fuzzy_set_dual', 'experiments'))
    assert(os.path.exists(exp_root_dir))

    stats_file = os.path.join(exp_root_dir, synset,
            'vanilla_100_centerize_relative', 'outputs',
            'complementarity_stats.txt')
    assert(os.path.exists(stats_file))

    stats = {}
    with open(stats_file, 'r') as f:
        lines = f.read().splitlines()
        for line in lines:
            key, value = line.split(': ')
            if key == '# Partial Shapes': value = int(value)
            else: value = float(value)
            stats[key] = value

    return stats


def generate_table(all_stats, out_file):
    all_synsets = all_stats.keys()
    all_synsets.sort()

    lines = ''

    line = '| {: ^24} |'.format('Category')
    for synset in all_synsets: line += ' {: ^10} |'.format(synset)
    line += ' {: ^10} |'.format('Mean')
    lines += (line + '\n')

    line = '| ------------------------ |'
    for synset in all_synsets: line += ' ---------- |'.format(synset)
    line += ' ---------- |'
    lines += (line + '\n')

    counts = {}
    total_counts = 0
    key = '# Partial Shapes'
    line = '| {: ^24} |'.format(key)
    for synset in all_synsets:
        counts[synset] = all_stats[synset][key]
        total_counts += counts[synset]
        line += ' {:10d} |'.format(counts[synset])
    line += ' {: ^10} |'.format('.')
    lines += (line + '\n')

    Mean = 0.
    total_value = 0.
    key = 'Recall@1'
    line = '| {: ^24} |'.format(key)
    for synset in all_synsets:
        value = all_stats[synset][key]
        total_value += (value * counts[synset])
        line += ' {:10.1f} |'.format(value)
    line += ' {:10.1f} |'.format(total_value / float(total_counts))
    lines += (line + '\n')

    Mean = 0.
    total_value = 0.
    key = 'Recall@10'
    line = '| {: ^24} |'.format(key)
    for synset in all_synsets:
        value = all_stats[synset][key]
        total_value += (value * counts[synset])
        line += ' {:10.1f} |'.format(value)
    line += ' {:10.1f} |'.format(total_value / float(total_counts))
    lines += (line + '\n')

    Mean = 0.
    total_value = 0.
    key = 'Median Percentile Rank'
    line = '| {: ^24} |'.format(key)
    for synset in all_synsets:
        value = all_stats[synset][key]
        total_value += (value * counts[synset])
        line += ' {:10.1f} |'.format(value)
    line += ' {:10.1f} |'.format(total_value / float(total_counts))
    lines += (line + '\n')

    Mean = 0.
    total_value = 0.
    key = 'Mean Percentile Rank'
    line = '| {: ^24} |'.format(key)
    for synset in all_synsets:
        value = all_stats[synset][key]
        total_value += (value * counts[synset])
        line += ' {:10.1f} |'.format(value)
    line += ' {:10.1f} |'.format(total_value / float(total_counts))
    lines += (line + '\n')


    print(lines)
    with open(out_file, 'w') as f: f.write(lines)
    print("Saved '{}'.".format(out_file))


if __name__ == '__main__':
    all_synsets = ['Airplane', 'Car', 'Chair', 'Guitar', 'Lamp', 'Rifle',
            'Sofa', 'Table', 'Watercraft']

    all_stats = {}
    for synset in all_synsets:
        all_stats[synset] = get_complementarity_stats(synset)

    out_name = os.path.splitext(os.path.basename(__file__))[0]
    out_file = os.path.join(BASE_DIR, '{}.txt'.format(out_name))
    generate_table(all_stats, out_file)

