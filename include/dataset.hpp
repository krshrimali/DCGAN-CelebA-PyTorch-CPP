//
//  dataset.hpp
//  DCGAN
//
//  Created by Kushashwa Ravi Shrimali on 21/08/19.
//  Copyright © 2019 Kushashwa Ravi Shrimali. All rights reserved.
//

#ifndef dataset_hpp
#define dataset_hpp

#include <iostream>
#include <opencv2/opencv.hpp>
#include <torch/torch.h>
#include <dirent.h>
#include <torch/script.h>

// Function to return image read at location given as type torch::Tensor
torch::Tensor read_data(std::string location, int resize);

// Function to return label from int (0, 1 for binary and 0, 1, ..., n-1 for n-class classification) as type torch::Tensor
torch::Tensor read_label(int label);

// Function returns vector of tensors (images) read from the list of images in a folder
std::vector<torch::Tensor> process_images(std::vector<std::string> list_images, int resize);

// Function returns vector of tensors (labels) read from the list of labels
std::vector<torch::Tensor> process_labels(std::vector<int> list_labels);

// Function to load data from given folder(s) name(s) (folders_name)
// Returns pair of vectors of string (image locations) and int (respective labels)
std::pair<std::vector<std::string>, std::vector<int>> load_data_from_folder(std::vector<std::string> folders_name);

// Function to train the network on train data
template<typename Dataloader>
void train(torch::jit::script::Module net, torch::nn::Linear lin, Dataloader& data_loader, torch::optim::Optimizer& optimizer, size_t dataset_size);

// Function to test the network on test data
template<typename Dataloader>
void test(torch::jit::script::Module network, torch::nn::Linear lin, Dataloader& loader, size_t data_size);

// Custom Dataset class
class CustomDataset : public torch::data::Dataset<CustomDataset> {
private:
    /* data */
    // Should be 2 tensors
    std::vector<torch::Tensor> states, labels;
    size_t ds_size;
public:
    CustomDataset(std::vector<std::string> list_images, std::vector<int> list_labels, int resize=224) {
        states = process_images(list_images, resize);
        labels = process_labels(list_labels);
        ds_size = states.size();
    };
    
    torch::data::Example<> get(size_t index) override {
        /* This should return {torch::Tensor, torch::Tensor} */
        torch::Tensor sample_img = states.at(index);
        torch::Tensor sample_label = labels.at(index);
        return {sample_img.clone(), sample_label.clone()};
    };
    
    // cv::Mat show_batch(int batch_size = 3) {
    //     /* 
    //     Visualize batch of data (by default 3x3) 
    //     */
    //     // Declare img_array as a pointer
    //     cv::Mat* img_varray;
    //     for(int i = 0; i < batch_size; i++) {
    //         *(img_varray + i) = get(i).at(0);
    //     }
    //     cv::Mat out;
    //     cv::hconcat(img_varray, 3, out);
    //     return out;
    // }

    void show_batch(int batch_size = 3) {
        /* 
        Visualize batch of data (by default 3x3) 
        */
        // Declare img_array as a pointer
        cv::Mat* img_varray = nullptr;
        for(int i = 0; i < batch_size; i++) {
            std::memcpy((void*)get(i).data.data_ptr(), (void*)*(img_varray + i)->data, sizeof(float) * get(i).data.numel());
        }
        cv::Mat out;
        cv::hconcat(img_varray, 3, out);
        // Save the image as out.jpg
        cv::imwrite("out.jpg", out);
        std::cout << "Image saved as out.jpg" << std::endl;
    }

    torch::optional<size_t> size() const override {
        return int(ds_size);
    };
};

#endif /* dataset_hpp */
