//
//  main.cpp
//  DCGAN
//
//  Created by Kushashwa Ravi Shrimali on 19/08/19.
//  Copyright © 2019 Kushashwa Ravi Shrimali. All rights reserved.
//

#include <iostream>
#include <torch/torch.h>

class Arguments {
public:
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
    Arguments(std::string data_root, int num_workers, int bs, int img_size, int num_channels, int length_latent_vector, int depth_feature_maps_g, int depth_feature_maps_d, int number_epochs, float learning_rate, float beta_1, int num_gpu) {
        dataroot = data_root;
        workers = num_workers;
        batch_size = bs;
        image_size = img_size;
        nc = num_channels;
        nz = length_latent_vector;
        ngf = depth_feature_maps_g;
        ndf = depth_feature_maps_d;
        num_epochs = number_epochs;
        lr = learning_rate;
        beta1 = beta_1;
        ngpu = num_gpu;
    };
};

int main(int argc, const char * argv[]) {
    Arguments args = Arguments("data/celeba", 2, 128, 64, 3, 100, 64, 64, 5, 0.0002, 0.5, 1);
    std::cout << args.batch_size << std::endl;
    return 0;
}
