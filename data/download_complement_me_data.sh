#!/bin/bash

# Download ShapeNet model component point cloud data (877M).
wget https://shapenet.cs.stanford.edu/media/minhyuk/ComplementMe/data/components.tgz
tar xzvf components.tgz
rm components.tgz

# Download ShapeNet model semanic part point cloud data (220M).
wget https://shapenet.cs.stanford.edu/media/minhyuk/ComplementMe/data/parts.tgz
tar xzvf parts.tgz
rm parts.tgz

