#pragma once 
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cuComplex.h>

namespace cuda_operations {
    __global__ void _2DConv(int  in_width, int  in_height,int filter_width, int filter_height, int stride, int padding,
                           void* input, void* filters, void* output);
    __global__ void _1DConv(int in_width, int filter_width, int stride, int padding, 
                           void* input, void* filters, void* output);
    __global__ void elementWiseMultiplyComplex(int width, int height, cuComplex* input, cuComplex* filters, cuComplex* output);
    // Frequency Domain operations 
    __global__ void _1D_DFT(int in_width, cuComplex* input, cuComplex* output);
    __global__ void _1D_IDFT(int in_width, cuComplex* input, cuComplex* output);

    __global__ void _1D_FFT(int width, int height, cuComplex* input, cuComplex* output);
    __global__ void _1D_IFFT(int width, int height, cuComplex* input, cuComplex* output);
    void _2D_FFTConv(int in_width, int in_height, int filter_width, int filter_height,
                           cuComplex* input, cuComplex* filters, cuComplex* output);
}