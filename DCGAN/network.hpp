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

class Generator : public torch.nn.Module {
private:
    int ngpu;
    torch.nn.Module.Sequential main;
public:
    Generator(int ngpu_) {
        ngpu = ngpu_;
        /* TODO: Need to define a Sequential Module similar to python snippet below */
        
        /* Python Snippet */
        /* TODO: Check if nn.ConvTranspose2d exists in C++ API, if not - implement */
        
        /*
         self.main = nn.Sequential(
             # input is Z, going into a convolution
             nn.ConvTranspose2d( nz, ngf * 8, 4, 1, 0, bias=False),
             nn.BatchNorm2d(ngf * 8),
             nn.ReLU(True),
             # state size. (ngf*8) x 4 x 4
             nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
             nn.BatchNorm2d(ngf * 4),
             nn.ReLU(True),
             # state size. (ngf*4) x 8 x 8
             nn.ConvTranspose2d( ngf * 4, ngf * 2, 4, 2, 1, bias=False),
             nn.BatchNorm2d(ngf * 2),
             nn.ReLU(True),
             # state size. (ngf*2) x 16 x 16
             nn.ConvTranspose2d( ngf * 2, ngf, 4, 2, 1, bias=False),
             nn.BatchNorm2d(ngf),
             nn.ReLU(True),
             # state size. (ngf) x 32 x 32
             nn.ConvTranspose2d( ngf, nc, 4, 2, 1, bias=False),
             nn.Tanh()
             # state size. (nc) x 64 x 64
         )
         */
    }
    
    // Forward function
    torch::Tensor forward(torch::Tensor input) {
        return main(input);
    }
}

#endif /* network_hpp */
