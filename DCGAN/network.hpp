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

// Generator consists: ConvTranspose2d layers, BatchNorm layers, ReLU Activations
// Input to Generator: latent vector, z, drawn from standard normal distribution
// Output is 3x64x64 RGB Image
class Generator : public torch::nn::Module {
private:
    std::string dataroot;
    int workers;
    int batch_size;
    int image_size;
    int nc;
    int nz;
    int ngf;
    int ndf;
    int num_epochs;
    float lr;
    float beta1;
    int ngpu;
public:
    torch::nn::Sequential main;
    Generator(std::string dataroot_ = "data/celeba", int workers_ = 2, int batch_size_ = 128, int image_size_ = 64, int nc_ = 3, int nz_ = 100, int ngf_ = 128, int ndf_ = 128, int num_epochs_ = 5, float lr_ = 0.0002, float beta1_ = 0.5, int ngpu_ = 1) {
        
        dataroot = dataroot_;
        workers = workers_;
        batch_size = batch_size_;
        image_size = image_size_;
        nc = nc_;
        nz = nz_;
        ngf = ngf_;
        ndf = ndf_;
        num_epochs = num_epochs_;
        lr = lr_;
        beta1 = beta1_;
        ngpu = ngpu_;
        
        main = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(100, ngf*8, 4).with_bias(false).transposed(true)),
            torch::nn::BatchNorm(ngf*8),
            torch::nn::Functional(torch::relu),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(ngf*8, ngf*4, 4).stride(2).padding(1).with_bias(false).transposed(true)),
            torch::nn::BatchNorm(ngf*4),
            torch::nn::Functional(torch::relu),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(ngf*4, ngf*2, 4).stride(2).padding(1).with_bias(false).transposed(true)),
            torch::nn::BatchNorm(ngf*2),
            torch::nn::Functional(torch::relu),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(ngf*2, ngf, 4).stride(2).padding(1).with_bias(false).transposed(true)),
            torch::nn::BatchNorm(ngf),
            torch::nn::Functional(torch::relu),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(ngf, 3, 4).stride(2).padding(1).with_bias(false).transposed(true)),
            torch::nn::Functional(torch::tanh)
                                    // torch::nn::Conv2d(torch::nn::Conv2dOptions(nz, ngf*8, 4).stride(1).padding(0).with_bias(false).transposed(true)),
                                    // torch::nn::BatchNorm(ngf*8),
                                    // torch::nn::Functional(torch::relu),
                                    // torch::nn::Conv2d(torch::nn::Conv2dOptions(ngf*8, ngf*4, 4).stride(2).padding(1).with_bias(false).transposed(true)),
                                    // torch::nn::BatchNorm(ngf*4),
                                    // torch::nn::Functional(torch::relu),
                                    // torch::nn::Conv2d(torch::nn::Conv2dOptions(ngf*4, ngf*2, 4).stride(2).padding(1).with_bias(false).transposed(true)),
                                    // torch::nn::BatchNorm(ngf*2),
                                    // torch::nn::Functional(torch::relu),
                                    // torch::nn::Conv2d(torch::nn::Conv2dOptions(ngf*2, ngf, 4).stride(2).padding(1).with_bias(false).transposed(true)),
                                    // torch::nn::BatchNorm(ngf),
                                    // torch::nn::Functional(torch::relu),
                                    // torch::nn::Conv2d(torch::nn::Conv2dOptions(ngf, nc, 4).stride(2).padding(1).with_bias(false).transposed(true)),
                                    // torch::nn::Functional(torch::tanh)
        );
    }
    
    torch::nn::Sequential main_func() {
        return main;
    }
    
//    // Forward function
//    torch::Tensor forward(torch::Tensor input) {
//        return main(input);
//    }
};

class Discriminator : public torch::nn::Module {
private:
    std::string dataroot;
    int workers;
    int batch_size;
    int image_size;
    int nc;
    int nz;
    int ngf;
    int ndf;
    int num_epochs;
    float lr;
    float beta1;
    int ngpu;
public:
    torch::nn::Sequential main;
    Discriminator(std::string dataroot_ = "data/celeba", int workers_ = 2, int batch_size_ = 64, int image_size_ = 64, int nc_ = 3, int nz_ = 100, int ngf_ = 128, int ndf_ = 128, int num_epochs_ = 10, float lr_ = 0.0002, float beta1_ = 0.5, int ngpu_ = 1) {
        
        dataroot = dataroot_;
        workers = workers_;
        batch_size = batch_size_;
        image_size = image_size_;
        nc = nc_;
        nz = nz_;
        ngf = ngf_;
        ndf = ndf_;
        num_epochs = num_epochs_;
        lr = lr_;
        beta1 = beta1_;
        ngpu = ngpu_;
        
        main = torch::nn::Sequential(
            torch::nn::Conv2d(torch::nn::Conv2dOptions(3, ndf, 4).stride(2).with_bias(false)),
            torch::nn::Functional(torch::leaky_relu, 0.2),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(ndf, ndf*2, 4).stride(2).padding(1).with_bias(false)),
            torch::nn::BatchNorm(ndf*2),
            torch::nn::Functional(torch::leaky_relu, 0.2),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(ndf*2, ndf*4, 4).stride(2).padding(1).with_bias(false)),
            torch::nn::BatchNorm(ndf*4),
            torch::nn::Functional(torch::leaky_relu, 0.2),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(ndf*4, ndf*8, 4).stride(2).padding(1).with_bias(false)),
            torch::nn::BatchNorm(ndf*8),
            torch::nn::Functional(torch::leaky_relu, 0.2),
            torch::nn::Conv2d(torch::nn::Conv2dOptions(ndf*8, 1, 4).stride(1).padding(0).with_bias(false)),
            torch::nn::Functional(torch::sigmoid)
        );
        // main = torch::nn::Sequential(
        //                             torch::nn::Conv2d(torch::nn::Conv2dOptions(nc, ndf, 4).stride(2).padding(1).with_bias(false)),
        //                             torch::nn::Functional(torch::leaky_relu, 0.2),
        //                             torch::nn::Conv2d(torch::nn::Conv2dOptions(ndf, ndf*2, 4).stride(2).padding(1).with_bias(false)),
        //                             torch::nn::BatchNorm(ndf*2),
        //                             torch::nn::Functional(torch::leaky_relu, 0.2),
        //                             torch::nn::Conv2d(torch::nn::Conv2dOptions(ndf*2, ndf*4, 4).stride(2).padding(1).with_bias(false)),
        //                             torch::nn::BatchNorm(ndf*4),
        //                             torch::nn::Functional(torch::leaky_relu, 0.2),
        //                             torch::nn::Conv2d(torch::nn::Conv2dOptions(ndf*4, ndf*8, 4).stride(2).padding(1).with_bias(false)),
        //                             torch::nn::BatchNorm(ndf*8),
        //                             torch::nn::Functional(torch::leaky_relu, 0.2),
        //                             torch::nn::Conv2d(torch::nn::Conv2dOptions(ndf*8, 1, 4).stride(1).padding(0).with_bias(false))
        // );
    }
    
    torch::nn::Sequential main_func() {
        return main;
    }
    
//    // Forward function
//    torch::Tensor forward(torch::Tensor input) {
//        return main(input);
//    }
};

#endif /* network_hpp */
