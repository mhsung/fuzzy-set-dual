# Minhyuk Sung (mhsung@cs.stanford.edu)
# March 2018

import os, sys
BASE_DIR = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(__file__))))


def merge_obj_files(in_obj_files, out_obj_file):
    # Install MeshLab.
    # Ubuntu: sudo apt-get install meshlab

    script_file = os.path.join(BASE_DIR, 'merge_meshes.mlx')

    cmd = 'meshlabserver -i ' + ' '.join(in_obj_files) + \
            ' -s ' + script_file + ' -o ' + out_obj_file
    print(cmd)
    os.system(cmd)

