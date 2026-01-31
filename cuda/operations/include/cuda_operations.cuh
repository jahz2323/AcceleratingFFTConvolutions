#pragma once 
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <cuComplex.h>
#include <cufft.h>


#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


namespace cuda_operations {
    // Spatial Domain operations
    __global__ void _2DConv(int  in_width, int  in_height,int filter_width, int filter_height, int stride, int padding,
                           void* input, void* filters, void* output);
    __global__ void _1DConv(int in_width, int filter_width, int stride, int padding, 
                           void* input, void* filters, void* output);

    // Frequency Domain operations 
    __global__ void _1D_DFT(int in_width, cuComplex* input, cuComplex* output);
    __global__ void _1D_IDFT(int in_width, cuComplex* input, cuComplex* output);

    __global__ void _1D_FFT(int width, int height, cuComplex* input, cuComplex* output);
    __global__ void _1D_IFFT(int width, int height, cuComplex* input, cuComplex* output);

    // Optimised Shared Memory FFT
    template <bool inverse, bool isRowWise>
    __global__ void OptimisedSharedMemory1DFFT(int n, cuComplex* input, cuComplex* output);


    void _2D_FFTConv(int in_width, int in_height, int filter_width, int filter_height,
                           cuComplex* input, cuComplex* filters, cuComplex* output);

    void _2DcuFFTConv(cufftHandle plan, int in_width, int in_height, int filter_width, int filter_height,
                           cuComplex* input, cuComplex* filters, cuComplex* output);
    
    void Optimised2DFFTConv(int in_width, int in_height, cuComplex* input, cuComplex* filters, cuComplex* output);
                           
    // GPUSFFT operations
    void _2D_GPUSFFTConv(); 
    void _1D_GPUSFFT(int width, int height, cuComplex* input, cuComplex* output);

    // Utility kernels
    template <bool IsStandard>
    __global__ void elementWiseMultiplyComplex(int width, int height, cuComplex* input, cuComplex* filters, cuComplex* output);
    __global__  void bitreversal(int width, int height, cuComplex* data);
    __global__ void bitreversal(int n, void* storage);
    __global__ void float2complex(int width, int height, float* input, cuComplex* output);
    __global__ void complex2float(int width, int height, cuComplex* input, float* output);
    __global__ void copy(float* odata, float* idata);
    __global__ void copyComplex(cuComplex* odata, cuComplex* idata);
    __global__ void scaleOutput(int width, int height, cuComplex* output, float scale);
    __global__ void naivetranspose(int width, int height, cuComplex* input, cuComplex* output);
}