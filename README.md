## Learning Fuzzy Set Representations of Partial Shapes on Dual Embedding Spaces

[Minhyuk Sung](http://mhsung.github.io), [Anastasia Dubrovina](http://web.stanford.edu/~adkarni/), [Vladimir G. Kim](http://vova.kim), and [Leonidas Guibas](https://geometry.stanford.edu/member/guibas/)<br>
Conditionally accepted to SGP 2018<br>
[[Project]](https://mhsung.github.io/fuzzy-set-dual.html) | [arXiv: TBA]

![teaser](https://mhsung.github.io/assets/images/fuzzy-set-dual/teaser.png)

### Citation
```
@article{Sung:2018,
  author = {Sung, Minhyuk and Dubrovina, Anastasia, and Kim, Vladimir G. and Guibas, Leonidas},
  title = {Learning Fuzzy Set Representations of Partial Shapes on Dual Embedding Spaces},
  Journal = {Computer Graphics Forum (Proc. of Symposium on Geometry Processing (SGP))}, 
  year = {2018}
}
```

### Introduction
This neural-network-based framework analyzes an uncurated collection of 3D models from the same category and learns two important types of semantic relations among full and partial shapes: complementarity and interchangeability. The former helps to identify which two partial shapes make a complete plausible object, and the latter indicates that interchanging two partial shapes from different objects preserves the object plausibility. These two relations are modeled as *fuzzy set* operations performed across the *dual* partial shape embedding spaces, and within each space, respectively, and *jointly* learned by encoding partial shapes as *fuzzy sets* in the dual spaces.

### Requirements
- Python-gflags (tested with ver. 3.1.2)
- Networkx (tested with ver. 2.1)
- Numpy (tested with ver. 1.14.2)
- Pandas (tested with ver. 0.23.0)
- Scipy (tested with ver. 1.0.1)
- TensorFlow-gpu (tested with ver. 1.4.0)

### Reproducing paper results
#### Data download
Download [ComplementMe](https://mhsung.github.io/component-assembly.html) component point cloud data:
```
cd data
./download_complement_me_data.sh
cd ..
```

#### Pretrained model download
We provide pretrained models for all categories. Run:
```
cd fuzzy_set_dual/experiments
./download_pretrained_models.sh
cd ../../
```

#### Network training
You can also easily train the network from scratch. Specify a category of shapes to train (set one of directory names in `data/components`. e.g. Chair. Case-sensitive):
```
export synset=Chair
```

Move to the experiment directory:
```
cd fuzzy_set_dual/experiments
```

For learning *complementarity*, run:
```
./run_experiment.py --relative --train
```

For learning *interchangeability*, run:
```
./run_experiment.py --train
```

The trained models are stored in `fuzzy_set_dual/experiments/($synset)/vanilla_100_centerize_relative` and `fuzzy_set_dual/experiments/($synset)/vanilla_100_centerize` directories, respectively.

The evaluations introduced in the paper are performed with the trained models when runnning the same `run_experiment.py` script without the `--train` option.

For running the evaluation code for all categories, run:
```
batch_run_complementarity.sh
batch_run_interchangeability.sh
```

#### Regenerating paper figures/tables.
We provide script files in [figures](figures) regenerating results in the paper figures and tables. The script files require [ComplementMe](https://mhsung.github.io/component-assembly.html) *mesh* data, and the mesh data is provided by the authors upon request (see [here](https://mhsung.github.io/component-assembly.html#data-download)). Download the mesh data in [data](data) directory and unzip them. The outputs are stored as mesh files, and the results of compared methods are not generated. Also, the position of retrieved complement partial shapes are not predicted since it is not a part of this project.

Note that the scripts have a dependency with [MeshLab](http://www.meshlab.net/). Ubuntu users can install with apt-get:
```
sudo apt-get install meshlab
```

### Acknowledgements
This project is developed based on the code in [ComplementMe](https://github.com/mhsung/complement-me). The files in [network_utils](network_utils) (except `resample_points.py`) are directly brought from the [PointNet++](https://github.com/charlesq34/pointnet2).

### License
This code is released under the MIT License. Refer to [LICENSE](LICENSE) for details.
