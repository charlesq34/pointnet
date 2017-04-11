#!/bin/bash

# Download original ShapeNetPart dataset (around 1GB)
wget https://shapenet.cs.stanford.edu/ericyi/shapenetcore_partanno_v0.zip
unzip shapenetcore_partanno_v0.zip
rm shapenetcore_partanno_v0.zip

# Download HDF5 for ShapeNet Part segmentation (around 346MB)
wget https://shapenet.cs.stanford.edu/media/shapenet_part_seg_hdf5_data.zip
unzip shapenet_part_seg_hdf5_data.zip
rm shapenet_part_seg_hdf5_data.zip

