PWLregions
=======

This repo contains code that can compute the piecewise affine (PWA) representation of a neural network. The algorithm does this by considering the layers of the network one by one, solving the resulting hyperplane arrangement problem separately for each previously found region. This is explained [here](https://arxiv.org/abs/1910.03879). This is a general procedure, and can be applied to any network with PWA activation functions and linear/affine layers (fully connected, convolutions, etc). The main limitation of this approach is that the runtime is exponential in both the number of input nodes and the depth of the network. An alternative algorithm developed is available at [this repo](https://github.com/95616ARG/SyReNN), which is less general (it only finds the regions on a 2D restriction of the input space), but is more efficient and numerically stable, as well as being suitable for use on a compute server.

The original code is written in MATLAB, despite most deep learning libraries being written in Python. This was done because MATLAB has access to the [MPT toolbox](https://www.mpt3.org/), which handles the necessary geometric computations. This toolbox also has methods for multiple-parametric optimisation and both implicit and explicit model predictive control (MPC), which is the direction we want to take this research. A conversion script for Tensorflow models is provided.

Install
-----
You will need MATLAB and Python. For MATLAB you will need to install the [MPT toolbox](https://www.mpt3.org/), which has instructions on its site. If you want to run MATLAB in Jupyter notebooks (see next section for instructions), you will need Python 3.7 specifically. This code has been tested with MATLAB 2020a and Python 3.7 only. 

The appropriate addpath()'s are called in the code. All you need to do is run `pwlregions_init.m`. This can be done manually every time you open MATLAB, or you can add the following line to your MATLAB `startup.m` file to make this permanent: 

```
run('path/to/install/pwlregions_init.m)
```

Running the notebooks
-----
Jupyter notebooks are an easy way to show the progression of your thinking, and to share your ideas with others. It turns out that Jupyter is general and you can install any language, as long as there is a kernel available. The notebooks can also be viewed on GitHub (assuming I've kept them up to date), so its fine to run MATLAB locally if you prefer.

First, clone this repo. Then, install [Anaconda](https://www.anaconda.com/) if you haven't already (both anaconda and miniconda are fine), and create a new environment with Python 3.7:

```bash
conda create -n pwa python=3.7 anaconda jupyter
conda activate pwa
```

This is done because the MATLAB engine only works with Python 2.7, 3.5, and 3.7. The code is only tested with Python 3.7. 

In this environment, you will need to install [this MATLAB kernel](https://github.com/Calysto/matlab_kernel) for Jupyter. This allows you to create a MATLAB notebook which can show plots etc.

Run the follow check that `matlab` has been added to the list of kernels in your environment:

```
jupyter kernelspec list
```

You should now be able to start Jupyter (I like `jupyter lab`) and create a new MATLAB notebook. The first command you execute will take a second to load up the MATLAB engine.


It is also possible to run Python in the same notebook using the %%python magic, but I haven't found a way to make them share environments/variables (this can be done with python and julia, as shown [here](https://github.com/binder-examples/julia-python/blob/master/python-and-julia.ipynb)). For now, if you want to run them side by side you'll have to make them communicate via files.




TODO
-----
+ Add comments
+ Update tests
+ Make a script that runs all tests
+ Replace pwa_matlab with a conversion script
+ Network class that has get_pwa_repr() as a method and can be initialised from json / matlab struct
+ Make a conversion script in python
+ Make a to_json() method for PWANetwork
+ Add examples in jupyter notebooks
+ Add support for basic layers such as batch norm, dropout, etc
+ Add support for convolutions and pooling
+ Find a way to make MATLAB and Python share variables through the MATLAB engine


Done
----
+ Add "install" instructions for MATLAB ( 10/8/2020 11:09 )
+ MATLAB in Jupyter notebooks ( 10/8/2020 11:00 )
+ Restructure repo ( 10/8/2020 9:50 )





