//
//  dataset.cpp
//  DCGAN
//
//  Created by Kushashwa Ravi Shrimali on 21/08/19.
//  Copyright © 2019 Kushashwa Ravi Shrimali. All rights reserved.
//

#include "dataset.hpp"

torch::Tensor read_data(std::string location) {
    /*
     Function to return image read at location given as type torch::Tensor
     Resizes image to (224, 224, 3)
     Parameters
     ===========
     1. location (std::string type) - required to load image from the location
     
     Returns
     ===========
     torch::Tensor type - image read as tensor
     */
    cv::Mat img = cv::imread(location, 1);
    cv::resize(img, img, cv::Size(224, 224), cv::INTER_CUBIC);
    torch::Tensor img_tensor = torch::from_blob(img.data, {img.rows, img.cols, 3}, torch::kByte);
    img_tensor = img_tensor.permute({2, 0, 1});
    return img_tensor.clone();
}

torch::Tensor read_label(int label) {
    /*
     Function to return label from int (0, 1 for binary and 0, 1, ..., n-1 for n-class classification) as type torch::Tensor
     Parameters
     ===========
     1. label (int type) - required to convert int to tensor
     
     Returns
     ===========
     torch::Tensor type - label read as tensor
     */
    torch::Tensor label_tensor = torch::full({1}, label);
    return label_tensor.clone();
}

std::vector<torch::Tensor> process_images(std::vector<std::string> list_images) {
    /*
     Function returns vector of tensors (images) read from the list of images in a folder
     Parameters
     ===========
     1. list_images (std::vector<std::string> type) - list of image paths in a folder to be read
     
     Returns
     ===========
     std::vector<torch::Tensor> type - Images read as tensors
     */
    std::vector<torch::Tensor> states;
    for(std::vector<std::string>::iterator it = list_images.begin(); it != list_images.end(); ++it) {
        torch::Tensor img = read_data(*it);
        states.push_back(img);
    }
    return states;
}

std::vector<torch::Tensor> process_labels(std::vector<int> list_labels) {
    /*
     Function returns vector of tensors (labels) read from the list of labels
     Parameters
     ===========
     1. list_labels (std::vector<int> list_labels) -
     
     Returns
     ===========
     std::vector<torch::Tensor> type - returns vector of tensors (labels)
     */
    std::vector<torch::Tensor> labels;
    for(std::vector<int>::iterator it = list_labels.begin(); it != list_labels.end(); ++it) {
        torch::Tensor label = read_label(*it);
        labels.push_back(label);
    }
    return labels;
}

std::pair<std::vector<std::string>,std::vector<int>> load_data_from_folder(std::vector<std::string> folders_name) {
    /*
     Function to load data from given folder(s) name(s) (folders_name)
     Returns pair of vectors of string (image locations) and int (respective labels)
     Parameters
     ===========
     1. folders_name (std::vector<std::string> type) - name of folders as a vector to load data from
     
     Returns
     ===========
     std::pair<std::vector<std::string>, std::vector<int>> type - returns pair of vector of strings (image paths) and respective labels' vector (int label)
     */
    std::vector<std::string> list_images;
    std::vector<int> list_labels;
    int label = 0;
    for(auto const& value: folders_name) {
        std::string base_name = value + "/";
        // cout << "Reading from: " << base_name << endl;
        DIR* dir;
        struct dirent *ent;
        if((dir = opendir(base_name.c_str())) != NULL) {
            while((ent = readdir(dir)) != NULL) {
                std::string filename = ent->d_name;
                if(filename.length() > 4 && filename.substr(filename.length() - 3) == "jpg") {
                    // cout << base_name + ent->d_name << endl;
                    // cv::Mat temp = cv::imread(base_name + "/" + ent->d_name, 1);
                    list_images.push_back(base_name + ent->d_name);
                    list_labels.push_back(label);
                }
            }
            closedir(dir);
        } else {
            std::cout << "Could not open directory" << std::endl;
            // return EXIT_FAILURE;
        }
        label += 1;
    }
    return std::make_pair(list_images, list_labels);
}


