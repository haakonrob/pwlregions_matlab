PWLregions
=======

This repo contains code that can compute the piecewise affine (PWA) representation of a neural network. The algorithm does this by considering the layers of the network one by one, solving the resulting hyperplane arrangement problem separately for each previously found region. This is explained [here](https://arxiv.org/abs/1910.03879). This is a general procedure, and can be applied to any network with PWA activation functions and linear/affine layers (fully connected, convolutions, etc). The main limitation of this approach is that the runtime is exponential in both the number of input nodes and the depth of the network. An alternative algorithm developed is available at [this repo](https://github.com/95616ARG/SyReNN), which is less general (it only finds the regions on a 2D restriction of the input space), but is more efficient and numerically stable, as well as being suitable for use on a compute server.

The original code is written in MATLAB, despite most deep learning libraries being written in Python. This was done because MATLAB has access to the [MPT toolbox](http://people.ee.ethz.ch/~mpt/2/), which handles the necessary geometric computations. This toolbox also has methods for multiple-parametric optimisation and both implicit and explicit model predictive control (MPC), which is the direction we want to take this research. A conversion script for Tensorflow models is provided.

Install
-----



TODO
-----
+ Add comments
+ Add "install" instructions for MATlAb
+ Make a script that runs all tests
+ Replace pwa_matlab with a conversion script
+ Network class that has pwa() as a method and can be initialised from python / matlab
+ Add examples in jupyter notebooks
+ Add support for basic layers such as batch norm, dropout, etc
+ Add support for convolutions and pooling


DONE
----
+ Restructure repo ( 10/8/2020 9:50 )





