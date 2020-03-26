//
//  dataset.cpp
//  DCGAN
//
//  Created by Kushashwa Ravi Shrimali on 21/08/19.
//  Copyright © 2019 Kushashwa Ravi Shrimali. All rights reserved.
//

#include "dataset.hpp"

// Function to return image read at location given as type torch::Tensor
// Resizes image to (224, 224, 3)
// Parameters
// ===========
// 1. location (std::string type) - required to load image from the location
// 2. resize (int type) - required to resize an image
//
// Returns
// ===========
// torch::Tensor type - image read as tensor
torch::Tensor read_data(std::string location, int resize=224) {
    cv::Mat img = cv::imread(location, 1);
    cv::resize(img, img, cv::Size(resize, resize), cv::INTER_CUBIC);
    torch::Tensor img_tensor = torch::from_blob(img.data, {img.rows, img.cols, 3}, torch::kByte);
    img_tensor = img_tensor.permute({2, 0, 1});
    return img_tensor.clone();
}

// Function to return label from int (0, 1 for binary and 0, 1, ..., n-1 for n-class classification) as type torch::Tensor
// Parameters
// ===========
// 1. label (int type) - required to convert int to tensor
//
// Returns
// ===========
// torch::Tensor type - label read as tensor
torch::Tensor read_label(int label) {
    torch::Tensor label_tensor = torch::full({1}, label);
    return label_tensor.clone();
}

//
// Function returns vector of tensors (images) read from the list of images in a folder
// Parameters
// ===========
// 1. list_images (std::vector<std::string> type) - list of image paths in a folder to be read
// 2. resize (int type) - argument for resizing each image
//
// Returns
// ===========
// std::vector<torch::Tensor> type - Images read as tensors
std::vector<torch::Tensor> process_images(std::vector<std::string> list_images, int resize=224) {
    std::vector<torch::Tensor> states;
    for(auto it = list_images.begin(); it != list_images.end(); ++it) {
        torch::Tensor img = read_data(*it, resize);
        states.push_back(img);
    }
    return states;
}

// Function returns vector of tensors (labels) read from the list of labels
// Parameters
// ===========
// 1. list_labels (std::vector<int> list_labels) -
//
// Returns
// ===========
// std::vector<torch::Tensor> type - returns vector of tensors (labels)
std::vector<torch::Tensor> process_labels(std::vector<int> list_labels) {
    std::vector<torch::Tensor> labels;
    for(auto it = list_labels.begin(); it != list_labels.end(); ++it) {
        torch::Tensor label = read_label(*it);
        labels.push_back(label);
    }
    return labels;
}

// Function to load data from given folder(s) name(s) (folders_name)
// Returns pair of vectors of string (image locations) and int (respective labels)
// Parameters
// ===========
// 1. folders_name (std::vector<std::string> type) - name of folders as a vector to load data from
//
// Returns
// ===========
// std::pair<std::vector<std::string>, std::vector<int>> type - returns pair of vector of strings (image paths) and respective labels' vector (int label)
std::pair<std::vector<std::string>,std::vector<int>> load_data_from_folder(std::vector<std::string> folders_name) {
    std::vector<std::string> list_images;
    std::vector<int> list_labels;

    assert(folders_name.size() != 0);

    int label = 0;
    for(auto const& value: folders_name) {
        std::string base_name;
        if(*value.rbegin() != '/') base_name = value + "/";
        else base_name = value;

        DIR* dir;
        struct dirent *ent;
        if((dir = opendir(base_name.c_str())) != NULL) {
            while((ent = readdir(dir)) != NULL) {
                std::string filename = ent->d_name;

                std::string file_extension = filename.substr(filename.find("."));
                assert(file_extension == ".jpg" || file_extension == ".png" || file_extension == ".jpeg");

                list_images.push_back(base_name + ent->d_name);
                list_labels.push_back(label);
            }
            closedir(dir);
        } else {
            std::cout << "Could not open directory " << base_name.c_str() << std::endl;
            exit(-1);
        }
        label += 1;
    }
    return std::make_pair(list_images, list_labels);
}