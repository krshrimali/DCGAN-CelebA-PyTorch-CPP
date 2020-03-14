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

// Function to return label from int (0, 1 for binary and 0, 1, ..., n-1 for n-class classification) as type 
// torch::Tensor
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
void train(torch::jit::script::Module net, torch::nn::Linear lin, Dataloader& data_loader, \
torch::optim::Optimizer& optimizer, size_t dataset_size);

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

    // Visualize batch of data (by default 3x3)
    void show_batch(int batch_size = 3) {
        cv::Mat* img_varray = new cv::Mat[batch_size*batch_size];
        for(int i = 0; i < batch_size*batch_size; i++) {
            torch::Tensor out_tensor = get(i).data.squeeze().detach().permute({1, 2, 0});
            out_tensor = out_tensor.clamp(0, 255).to(torch::kCPU).to(torch::kU8);
            *(img_varray + i) = cv::Mat::eye(out_tensor.sizes()[0], out_tensor.sizes()[1], CV_8UC3);
            std::memcpy((img_varray + i)->data, out_tensor.data_ptr(), sizeof(torch::kU8) * out_tensor.numel());
        }
        cv::Mat out(256, 256, CV_8UC3);
        cv::Mat temp_out(256, 256, CV_8UC3);
        for(int vconcat_times = 0; vconcat_times < batch_size; vconcat_times++) {
            cv::cvtColor(*(img_varray + vconcat_times*batch_size), out, cv::COLOR_BGR2RGB);
            for(int hconcat_times = vconcat_times*batch_size; hconcat_times < (vconcat_times+1)*batch_size - 1; hconcat_times++) {
                cv::cvtColor(*(img_varray + hconcat_times + 1), *(img_varray + hconcat_times + 1), cv::COLOR_BGR2RGB);
                cv::hconcat(out, *(img_varray + hconcat_times + 1), out);
            }
            if(vconcat_times == 0)
                temp_out = out;
            else {
                cv::vconcat(temp_out, out, temp_out);
            }
        }
        cv::cvtColor(temp_out, temp_out, cv::COLOR_BGR2RGB);
        cv::imwrite("out.jpg", temp_out);
        std::cout << "Image saved as out.jpg" << std::endl;
    }

    // Visualizes sample at the given index
    void show_sample(int index) {
        torch::Tensor out_tensor_ = get(index).data.squeeze().detach().permute({1, 2, 0});
        out_tensor_ = out_tensor_.clamp(0, 255).to(torch::kCPU).to(torch::kU8);
        cv::Mat sample_img(out_tensor_.sizes()[0], out_tensor_.sizes()[1], CV_8UC3);
        std::memcpy(sample_img.data, out_tensor_.data_ptr(), sizeof(torch::kU8) * out_tensor_.numel());
        cv::imwrite("sample.jpg", sample_img);
        std::cout << "Image saved as sample.jpg" << std::endl;
    }

    torch::optional<size_t> size() const override {
        return int(ds_size);
    };
};

#endif /* dataset_hpp */