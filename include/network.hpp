//
//  network.hpp
//  DCGAN
//
//  Created by Kushashwa Ravi Shrimali on 23/08/19.
//  Copyright © 2019 Kushashwa Ravi Shrimali. All rights reserved.
//

#ifndef network_hpp
#define network_hpp

#include <iostream>
#include <torch/torch.h>
#include <opencv2/opencv.hpp>
#include <torch/script.h>

// DCGAN uses convolutional in Discriminator
// And convolutional-transpose layers in Generator

// TODO: This has been solved in Nightly Version, update when 1.5 is released (stable)
struct ConvTranspose2dWrapperImpl : public torch::nn::ConvTranspose2dImpl {
  using torch::nn::ConvTranspose2dImpl::ConvTranspose2dImpl;

  torch::Tensor forward(const torch::Tensor& input) {
    return torch::nn::ConvTranspose2dImpl::forward(input, c10::nullopt);
  }
};

TORCH_MODULE(ConvTranspose2dWrapper);


// Generator consists: ConvTranspose2d layers, BatchNorm layers, ReLU Activations
// Input to Generator: latent vector, z, drawn from standard normal distribution
// Output is 3x64x64 RGB Image
class Generator : public torch::nn::Module {
  private:
    int nc;
    int nz;
    int ngf;
    int ndf;
    torch::nn::Sequential main;
  public:
    Generator(int nc_ = 3, int nz_ = 100, int ngf_ = 64) {
      nc = nc_;
      nz = nz_;
      ngf = ngf_;

      // TODO: ConvTranspose2dWrapper will be replaced with ConvTranspose2d in Libtorch 1.5 Version
      main = torch::nn::Sequential(
        ConvTranspose2dWrapper(torch::nn::ConvTranspose2dOptions(nz, ngf*8, 4).stride(1).padding(0).bias(false)),
        torch::nn::BatchNorm2d(ngf*8),
        torch::nn::Functional(torch::relu),
        ConvTranspose2dWrapper(torch::nn::ConvTranspose2dOptions(ngf*8, ngf*4, 4).stride(2).padding(1).bias(false)),
        torch::nn::BatchNorm2d(ngf*4),
        torch::nn::Functional(torch::relu),
        ConvTranspose2dWrapper(torch::nn::ConvTranspose2dOptions(ngf*4, ngf*2, 4).stride(2).padding(1).bias(false)),
        torch::nn::BatchNorm2d(ngf*2),
        torch::nn::Functional(torch::relu),
        ConvTranspose2dWrapper(torch::nn::ConvTranspose2dOptions(ngf*2, ngf, 4).stride(2).padding(1).bias(false)),
        torch::nn::BatchNorm2d(ngf),
        torch::nn::Functional(torch::relu),
        ConvTranspose2dWrapper(torch::nn::ConvTranspose2dOptions(ngf, nc, 4).stride(2).padding(1).bias(false)),
        torch::nn::Functional(torch::tanh)
      );
    }

    torch::nn::Sequential get_module() {
      return main;
    }    
};

class Discriminator : public torch::nn::Module {
  private:
    int nc;
    int nz;
    int ngf;
    int ndf;
    torch::nn::Sequential main;
  public:
    Discriminator(int nc_ = 3, int ngf_ = 64, int ndf_ = 64) {
      nc = nc_;
      ndf = ndf_;
      
      main = torch::nn::Sequential(
        torch::nn::Conv2d(torch::nn::Conv2dOptions(nc, ndf, 4).stride(2).padding(1).bias(false)),
        torch::nn::Functional(torch::leaky_relu, 0.2),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(ndf, ndf*2, 4).stride(2).padding(1).bias(false)),
        torch::nn::BatchNorm2d(ndf*2),
        torch::nn::Functional(torch::leaky_relu, 0.2),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(ndf*2, ndf*4, 4).stride(2).padding(1).bias(false)),
        torch::nn::BatchNorm2d(ndf*4),
        torch::nn::Functional(torch::leaky_relu, 0.2),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(ndf*4, ndf*8, 4).stride(2).padding(1).bias(false)),
        torch::nn::BatchNorm2d(ndf*8),
        torch::nn::Functional(torch::leaky_relu, 0.2),
        torch::nn::Conv2d(torch::nn::Conv2dOptions(ndf*8, 1, 4).stride(1).padding(0).bias(false)),
        torch::nn::Functional(torch::sigmoid)
      );
    }

    torch::nn::Sequential get_module() {
      return main;
    }
};

#endif /* network_hpp */
