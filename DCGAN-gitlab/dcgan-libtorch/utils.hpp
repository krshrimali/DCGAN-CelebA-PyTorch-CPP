#pragma once

// utils.hpp
// This file contains all the utility functions I need. Ranging from printing a vector, to printing a tensor of ND (N <= 3)
// DCGAN
//
// Created by Kushashwa Ravi Shrimali on 19/02/2020
// Copyright 2019 Kushashwa Ravi Shrimali. All rights reserved.
//

#include <iostream>
#include <vector>
#include <type_traits>

template <class T>
void printVector(T self) {
    typedef typename T::value_type value_type;

}
