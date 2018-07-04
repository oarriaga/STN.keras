# Spatial transformer networks
Implementation of spatial transformer networks in keras 2.1.6 using tensorflow 1.8.0 as backend.

![alt tag](images/transformation.png)

![alt tag](images/results.jpg)

# A few notes
* This project only implements the 2D affine transformation as defined in equation (1) in https://arxiv.org/abs/1506.02025
* Both input and output coordinates are normalized between -1 and 1. The origin is located at the center of the image.
* In `mnist_cluttered_example.ipynb`, the spatial transformer is connected to a classifer network, and the two networks are jointly trained. If you decide to train the spatial transformer alone as an object detector, using an L1 loss between the transformed image and the ground truth may not be a good idea. Try to represent the ground truth in the form of an affine transformation matrix.
