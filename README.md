# DCGAN-CelebA-PyTorch-CPP

DCGAN Implementation (on CelebA dataset) using PyTorch C++ Frontend API (Libtorch)

- Training Code location: DCGAN/main.cpp
- Generator and Discriminator Definition: DCGAN/network.hpp
- Dataset Class: DCGAN/dataset.hpp and DCGAN/dataset.cpp
- Tested on Libtorch Version: Stable 1.4.0 (cxx11 ABI) with and without CUDA (10.1), Linux

How is this different from dcgan sample of PyTorch?

1. This loads a custom dataset (which is not in the dataset class of PyTorch) - CelebA.
2. Since some users prefer using Sequential Modules, so this example uses Sequential Module.

Utility Functions (to visualize images & create animation), and architecture is inherited from the PyTorch Example on DCGAN (https://github.com/pytorch/examples/blob/master/cpp/dcgan/).

Please note that this is in no way targeted to achieve a certain accuracy, but only focuses on creating an example template for DCGAN using Libtorch on CelebA Dataset.
