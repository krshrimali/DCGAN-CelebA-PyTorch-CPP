# DCGAN-CelebA-PyTorch-CPP

DCGAN Implementation (on CelebA dataset) using PyTorch C++ Frontend API (Libtorch)

<<<<<<< HEAD
- Training Code location: `src/main.cpp`
- Generator and Discriminator Definition: `include/network.hpp`
- Dataset Class: `include/dataset.hpp` and `src/dataset.cpp`
=======
- Training Code location: src/main.cpp
- Generator and Discriminator Definition: include/network.hpp
- Dataset Class: include/dataset.hpp and src/dataset.cpp
>>>>>>> f822b02e2d52084adb7c7d752f612b87ca688782
- Tested on Libtorch Version: Stable 1.4.0 (cxx11 ABI) with and without CUDA (10.1), Linux

How is this different from dcgan sample of PyTorch?

1. This loads a custom dataset (which is not in the dataset class of PyTorch) - CelebA.
2. Since some users prefer using Sequential Modules, so this example uses Sequential Module.

Utility Functions (to visualize images & create animation), and architecture is inherited from the PyTorch Example on DCGAN (https://github.com/pytorch/examples/blob/master/cpp/dcgan/).

Please note that this is in no way targeted to achieve a certain accuracy, but only focuses on creating an example template for DCGAN using Libtorch on CelebA Dataset.

## Steps to Follow

1. Create a build directory: `mkdir build/`
2. Configure your CMake: `cmake -DCMAKE_PREFIX_PATH=<absolute path to libtorch> ..`
3. Create an output directory (in the build directory) to store the results & save checkpoints: `mkdir output/`
4. Build your project: `make`
5. Execute: `./bin/example/`
6. The saved checkpoints and outputs will be at `output/` directory
7. To visualize, go back to the main directory: `cd ../`
8. Execute: `python3 utils/display_samples.py`
9. The outputs will be stores in `build/output/output_images/` directory
10. If you want to make an animation, run: `python3 utils/visualize.py` and it will save the animation for you in `build/output/output_animation/` directory

## Blog

<<<<<<< HEAD
Find more about DCGAN on my blog here: https://krshrimali.github.io/DCGAN-using-PyTorch-CPP/
=======
Find more about DCGAN on my blogs here:
1. https://krshrimali.github.io/DCGAN-using-PyTorch-CPP/
2. <to be added>  `TODO`
>>>>>>> f822b02e2d52084adb7c7d752f612b87ca688782

## Results

This is the output from random noise (batch of) images after ~10 epochs of training:

<img src="images/dcgan-output.png"/>

Happy learning!
