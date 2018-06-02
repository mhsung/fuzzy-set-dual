## Learning Fuzzy Set Representations of Partial Shapes on Dual Embedding Spaces

[Minhyuk Sung](http://mhsung.github.io), [Anastasia Dubrovina](http://web.stanford.edu/~adkarni/), [Vladimir G. Kim](http://vova.kim), and [Leonidas Guibas](https://geometry.stanford.edu/member/guibas/)<br>
Conditionally accepted to SGP 2018<br>
[[Project]](https://mhsung.github.io/fuzzy-set-dual.html) | [arXiv: TBA]

![teaser](https://mhsung.github.io/assets/images/fuzzy-set-dual/teaser.png)

### Citation
```
@article{Sung:2018,
  author = {Sung, Minhyuk and Dubrovina, Anastasia, and Kim, Vladimir G.
    and Guibas, Leonidas},
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
#### Downloading data
#### Training a network
#### Downloading pretrained models.
#### Regenerating paper figures/tables.



### Acknowledgements
This project is developed based on the code in [ComplementMe](https://github.com/mhsung/complement-me).

Also, the files in [network_utils](network_utils) except 'resample_points.py' are directly brought from the [PointNet++](https://github.com/charlesq34/pointnet2).

### License
This code is released under the MIT License. Refer to [LICENSE](LICENSE) for details.

### To-Do
- [ ] Script files reproducing results in the paper.
