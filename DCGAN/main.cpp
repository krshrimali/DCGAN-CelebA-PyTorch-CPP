//
//  main.cpp
//  DCGAN
//
//  Created by Kushashwa Ravi Shrimali on 19/08/19.
//  Copyright © 2019 Kushashwa Ravi Shrimali. All rights reserved.
//

#include <iostream>
#include <torch/torch.h>
#include "network.hpp"
#include "dataset.hpp"
#include "utils.hpp"

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

int ngf = 64;
int ndf = 64;

  torch::nn::Sequential netG(
      ConvTranspose2dWrapper(torch::nn::ConvTranspose2dOptions(300, ngf*8, 4).stride(1).padding(0).bias(false)),
      torch::nn::BatchNorm(ngf*8),
      torch::nn::Functional(torch::relu),
      ConvTranspose2dWrapper(torch::nn::ConvTranspose2dOptions(ngf*8, ngf*4, 4).stride(2).padding(1).bias(false)),
      torch::nn::BatchNorm(ngf*4),
      torch::nn::Functional(torch::relu),
      ConvTranspose2dWrapper(torch::nn::ConvTranspose2dOptions(ngf*4, ngf*2, 4).stride(2).padding(1).bias(false)),
      torch::nn::BatchNorm(ngf*2),
      torch::nn::Functional(torch::relu),
      ConvTranspose2dWrapper(torch::nn::ConvTranspose2dOptions(ngf*2, ngf, 4).stride(2).padding(1).bias(false)),
      torch::nn::BatchNorm(ngf),
      torch::nn::Functional(torch::relu),
      ConvTranspose2dWrapper(torch::nn::ConvTranspose2dOptions(ngf, 3, 4).stride(2).padding(1).bias(false)),
      torch::nn::Functional(torch::tanh)
      );

  torch::nn::Sequential netD(
      torch::nn::Conv2d(torch::nn::Conv2dOptions(3, ndf, 4).stride(2).padding(1).bias(false)),
      torch::nn::Functional(torch::leaky_relu, 0.2),
      torch::nn::Conv2d(torch::nn::Conv2dOptions(ndf, ndf*2, 4).stride(2).padding(1).bias(false)),
      torch::nn::BatchNorm2d(ndf*2),
      torch::nn::Functional(torch::leaky_relu, 0.2),
      torch::nn::Conv2d(torch::nn::Conv2dOptions(ndf*2, ndf*4, 4).stride(2).padding(1).bias(false)),
      torch::nn::BatchNorm2d(ndf*4),
      torch::nn::Functional(torch::leaky_relu, 0.2),
      torch::nn::Conv2d(torch::nn::Conv2dOptions(ndf*4, ndf*8, 4).stride(2).padding(1).bias(false)),
      torch::nn::BatchNorm(ndf*8),
      torch::nn::Functional(torch::leaky_relu, 0.2),
      torch::nn::Conv2d(torch::nn::Conv2dOptions(ndf*8, 1, 4).stride(1).padding(0).bias(false)),
      torch::nn::Functional(torch::sigmoid)
      );

  int main(int argc, const char * argv[]) {
    // dataroot, workers, batch_size, image_size, nc, nz, ngf, ndf, num_epochs, lr, beta1, ngpu
    Arguments args = Arguments("/home/kshrimali/Documents/DCGAN-cuda/dcgan-libtorch/data", 2, 64, 64, 3, 300, 64, 64, 5, 0.001, 0.5, 1);
    std::string images_name = args.dataroot + "/train";

    std::vector<std::string> folders_name;
    folders_name.push_back(images_name);

    // Get paths of images and labels as int from the folder paths
    std::pair<std::vector<std::string>, std::vector<int>> pair_images_labels = load_data_from_folder(folders_name);
    std::vector<std::string> list_images = pair_images_labels.first;
    std::vector<int> list_labels = pair_images_labels.second;

    // Originally 224 size, resize to 64 instead 
    auto custom_dataset = CustomDataset(list_images, list_labels, 64).map(torch::data::transforms::Normalize<>(0.5, 0.5)).map(torch::data::transforms::Stack<>());

    auto data_loader = torch::data::make_data_loader<torch::data::samplers::RandomSampler>(std::move(custom_dataset), torch::data::DataLoaderOptions().batch_size(args.batch_size).workers(2));

    torch::Device device = torch::kCPU;
    if(torch::cuda::is_available()) {
      device = torch::kCUDA;
    }

    std::cout << "Using device: " << device << std::endl; 
    Generator g = Generator();
    Discriminator d = Discriminator();

    // torch::nn::Sequential netG = g.main;
    // torch::nn::Sequential netD = d.main;
    netG->to(device);
    netD->to(device);

    torch::optim::Adam optimizerG(
        netG->parameters(), torch::optim::AdamOptions(2e-4).beta1(0.5));
    torch::optim::Adam optimizerD(
        netD->parameters(), torch::optim::AdamOptions(2e-4).beta1(0.5));

    int printEveryCheckpoint = 10;
    bool restoreFromCheckpoint = false; // set to false if you don't want to restore from checkpoint saved earlier
    if(restoreFromCheckpoint) {
      std::cout << "restoring from checkpoint..." << std::endl;
      torch::load(netG, "generator-checkpoint.pt");
      torch::load(optimizerG, "generator-optimizer-checkpoint.pt");
      torch::load(netD, "discriminator-checkpoint.pt");
      torch::load(optimizerD, "discriminator-optimizer-checkpoint.pt");
      std::cout << "restoring done" << std::endl;
    }
  
    int64_t checkpoint_counter = 1;
    auto options = torch::TensorOptions().device(device).requires_grad(false);
    for(int64_t epoch=1; epoch<=50; ++epoch) {
      int64_t batch_index=0;
      for(torch::data::Example<>& batch: *data_loader) {
        netD->zero_grad();
        torch::Tensor real_images = batch.data.to(device);
        torch::Tensor real_labels = torch::empty(batch.data.size(0), device).uniform_(1.0, 1.0);
        // Convert torch::tensor to cv::Mat and show the image

        torch::Tensor real_output = netD->forward(real_images);
        torch::Tensor d_loss_real = torch::binary_cross_entropy(real_output, real_labels);
        d_loss_real.backward();

        // Train discriminator with fake images
        torch::Tensor noise = torch::randn({batch.data.size(0), args.nz, 1, 1}, device);
        torch::Tensor fake_images = netG->forward(noise);
        torch::Tensor fake_labels = torch::zeros(batch.data.size(0), device);
        torch::Tensor fake_output = netD->forward(fake_images.detach());
        torch::Tensor d_loss_fake = torch::binary_cross_entropy(fake_output, fake_labels);
        d_loss_fake.backward();

        torch::Tensor d_loss = d_loss_real + d_loss_fake;
        optimizerD.step();

        // Train generator
        netG->zero_grad();
        fake_labels.fill_(1);
        fake_output = netD->forward(fake_images);
        torch::Tensor g_loss = torch::binary_cross_entropy(fake_output, fake_labels);
        g_loss.backward();
        optimizerG.step();

        std::cout << "Epoch: " << epoch << ", D_loss: " << d_loss.item<float>() << ", G_loss: " << g_loss.item<float>() << std::endl;
        batch_index++;
        // check point for every printEveryCheckpoint batches
        if(batch_index % printEveryCheckpoint == 0) {
          torch::save(netG, "color/generator-checkpoint.pt");
          torch::save(optimizerG, "color/generator-optimizer-checkpoint.pt");
          torch::save(netD, "color/discriminator-checkpoint.pt");
          torch::save(optimizerD, "color/discriminator-optimizer-checkpoint.pt");
          torch::Tensor samples = netG->forward(torch::randn({64, args.nz, 1, 1}, options));
          torch::save(samples, torch::str("color/dcgan-sample-", ++checkpoint_counter, ".pt"));
	  std::cout << "\n-> checkpoint " << ++checkpoint_counter << "\n";
        }
      }
    }
    return 0;
  }
