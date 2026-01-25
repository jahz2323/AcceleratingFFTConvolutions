#pragma once 

#include <chrono>
#include <iostream>
#include <torch/torch.h>
#include <math_constants.h>
#include <utils.cuh>
#include <cuda_operations.cuh>
using namespace cuda_operations;

class convolution_test {
public:
    static void test1DConvolution();
    static void test2DConvolution(int, int, int, int, int, int);
    static void testFFTConvolution();
    static void convolve();
};