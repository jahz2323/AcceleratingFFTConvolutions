#pragma once 
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cuComplex.h>
#include <math_constants.h>
namespace utils{
    __global__  void bitreversal(int width, int height, cuComplex* data);
    __global__ void bitreversal(int n, void* storage);
    __global__ void float2complex(int width, int height, float* input, cuComplex* output);
    __global__ void complex2float(int width, int height, cuComplex* input, float* output);
    __global__ void copy(float* odata, float* idata);
    __global__ void copyComplex(cuComplex* odata, cuComplex* idata);
    // __global__ void zeroPad2D(int in_width, int in_height, int out_width, int out_height, void* input, void* output);
    // __global__ void zeroPad1D(int in_width, int out_width, void* input, void* output);
    // __device__ int nextPowerOfTwo(int n);
    __global__ void naivetranspose(int width, int height, cuComplex* input, cuComplex* output);
    
}