#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <string>
#include <vector>

#include "cuda_operations.cuh"
#include "utils.cuh"


/**
 * @file cuda_operations.cu
 * @brief CUDA implementation of convolution operations 
    CUDA Conv 
    Torch Conv
    FFTConv 
    FFTConvOva
    FFTW Conv 
    cusFFT Conv 
**/

/**
    * @brief 2D Convolution CUDA kernel
*/
__global__ void cuda_operations::_2DConv(int in_width, int in_height, int filter_width, int filter_height, int stride, int padding,
                           void* input, void* filters, void* output)                   
{
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    auto idy = blockIdx.y * blockDim.y + threadIdx.y;
    int output_width = ((in_width - filter_width + 2 * padding) / stride) + 1;
    int output_height = ((in_height - filter_height + 2 * padding) / stride) + 1;
    // bounds check
    if  (idx >= output_width || idy >= output_height) return;
    
    float pvalue = 0.0f;
    // y[n,m] = sum(0,k) sum(0,l) w[k,l] * x[n-k,m-l]
    // dow (row,col)
    for (int k = 0; k < filter_height; k++) {
        for (int l = 0; l < filter_width; l++) {
            int n_start_pos = idy * stride - k + padding;
            int m_start_pos = idx * stride - l + padding;
            if (n_start_pos >= 0 && n_start_pos < in_height &&
                m_start_pos >= 0 && m_start_pos < in_width) {
                float x_value = static_cast<float*>(input)[n_start_pos * in_width + m_start_pos];
                float w_value = static_cast<float*>(filters)[k * filter_width + l];
                pvalue += x_value * w_value;
            }
        }
    }
    static_cast<float*>(output)[idy * output_width + idx] = pvalue;
}

/**
    * @brief 1D Convolution CUDA kernel
    Y[n] = sum(0,k) w[k] * x[n-k] 
    TODO: Realignment to match coalesced memory access
    TODO: Shared Memory optimization - load input, filter to shared mem 
*/
__global__ void cuda_operations::_1DConv(int in_width, int filter_width, int stride, int padding, void* input, void* filters, void* output) {
        auto idx = blockIdx.x * blockDim.x + threadIdx.x;
        if(idx >= ((in_width - filter_width + 2 * padding) / stride) + 1) return;
        float pvalue = 0.0f;

        // y[n] = sum(0,k) w[k] * x[n-k]
        for(int k =0; k< filter_width; k++){
            int n_start_pos = idx* stride - k + padding;
            if(n_start_pos >=0 && n_start_pos < in_width){
                float x_value = static_cast<float*>(input)[n_start_pos];
                float w_value = static_cast<float*>(filters)[k];
                pvalue += x_value * w_value;
            }
        }
        static_cast<float*>(output)[idx] = pvalue;
}

/**
    @brief Element-wise multiplication of complex numbers CUDA kernel
    @param width: Width of the input signal int
    @param height: Height of the input signal int
    @param input: Pointer to input signal (complex numbers)
    @param filters: Pointer to filter signal (complex numbers)
    @param output: Pointer to output signal (complex numbers)
*/
__global__ void cuda_operations::elementWiseMultiplyComplex(int width, int height, cuComplex* input, cuComplex* filters, cuComplex* output){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if(idx >= width || idy >= height) return;

    int index = idy * width + idx;
    cuComplex input_val = (input)[index];
    cuComplex filter_val = (filters)[index];
    cuComplex product = cuCmulf(input_val, filter_val);
    (output)[index] = product;
}

/**
    * @brief 1D DFT CUDA kernel
    @param in_width: Width of the input signal int 
    @param input: Pointer to input signal (complex numbers)
    @param output: Pointer to output signal (complex numbers)
*/
__global__ void cuda_operations::_1D_DFT(int in_width, cuComplex* input, cuComplex* output) {
    // Implementation of 1D DFT kernel
    // /x[k] = sum(0,N-1) x[n] * exp(-2pi * i * n * k / N)
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= in_width) return;
    cuComplex sum = make_cuComplex(0.0f, 0.0f);
    for(int n = 0; n < in_width; n++){
        float angle = -2.0f * 3.14159265359f * n * idx / in_width;
        cuComplex w = make_cuComplex(cosf(angle), sinf(angle));
        sum = cuCaddf(sum, cuCmulf(input[n], w));
    }
    output[idx] = sum;
}
/**
    * @brief 1D IDFT CUDA kernel
    @param in_width: Width of the input signal int 
    @param input: Pointer to input signal (complex numbers)
    @param output: Pointer to output signal (complex numbers)
*/
__global__ void cuda_operations::_1D_IDFT(int in_width, cuComplex* input, cuComplex* output) {
    // Implementation of 1D IDFT kernel
    // x[n] = (1/N) * sum(0,N-1) x[k] * exp(2pi * i * n * k / N)
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= in_width ) return;
    cuComplex sum = make_cuComplex(0.0f, 0.0f);

    for(int k = 0; k < in_width; k++){
        float angle = 2.0f * 3.14159265359f  * idx * k / in_width; // check if macro is available 
        cuComplex w = make_cuComplex(cosf(angle), sinf(angle));
        sum = cuCaddf(sum, cuCmulf(input[k], w));
    }

    float scale = 1.0f / in_width;
    output[idx] = make_cuComplex(sum.x  * scale, sum.y * scale);
}

/**
    @brief 1D FFT CUDA kernel
    handler for 1DFFT rows and columns
    @param width: Width of the input signal int
    @param height: Height of the input signal int
    @param input: Pointer to input signal (complex numbers)
    @param output: Pointer to output signal (complex numbers)
*/
__global__ void cuda_operations::_1D_FFT(int width, int height, cuComplex* input, cuComplex* output){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if(row >= height) return;

    cuComplex* row_data = input + row * width;
    
    // FFT computation
    int stages = static_cast<int>(log2f((float)width));
    for(int s = 1; s <= stages; s++){
        int m = 1 << s; // 2^s
        cuComplex wm = make_cuComplex(cosf(-2.0f * 3.14159265359f / m), sinf(-2.0f * 3.14159265359f / m));
        for(int k = 0; k < width; k += m){
            cuComplex w = make_cuComplex(1.0f, 0.0f);
            for(int j = 0; j < m / 2; j++){
                cuComplex t = cuCmulf(w, row_data[k + j + m / 2]);
                cuComplex u = row_data[k + j];
                row_data[k + j] = cuCaddf(u, t);
                row_data[k + j + m / 2] = cuCsubf(u, t);
                w = cuCmulf(w, wm);
            }
        }
    }
}

/**
    @brief 1D IFFT CUDA kernel
    handler for 1DIFFT rows and columns
    @param width: Width of the input signal int
    @param height: Height of the input signal int
    @param input: Pointer to input signal (complex numbers)
    @param output: Pointer to output signal (complex numbers)
*/
__global__ void cuda_operations::_1D_IFFT(int width, int height, cuComplex* input, cuComplex* output){
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if(row >= height) return;

    cuComplex* row_data = input + row * width;
    
    // IFFT computation
    for(int s = 1; s <= log2f(width); s++){
        int m = 1 << s; // 2^s
        cuComplex wm = make_cuComplex(cosf(2.0f * 3.14159265359f / m), sinf(2.0f * 3.14159265359f / m));
        for(int k = 0; k < width; k += m){
            cuComplex w = make_cuComplex(1.0f, 0.0f);
            for(int j = 0; j < m / 2; j++){
                cuComplex t = cuCmulf(w, row_data[k + j + m / 2]);
                cuComplex u = row_data[k + j];
                row_data[k + j] = cuCaddf(u, t);
                row_data[k + j + m / 2] = cuCsubf(u, t);
                w = cuCmulf(w, wm);
            }
        }
    }
    // Normalize the result
    for(int i = 0; i < width; i++){
        row_data[i] = make_cuComplex(row_data[i].x / width, row_data[i].y / width);
    }
}


/**
    * @brief 2D FFTConv CUDA kernel
    Take input and filter as Complex* device pointers 
    take input dimensions and filter dimensions
    output the convolved result to output pointers as Complex*, 
    Once in the frequency domain, perform element-wise multiplication
    then compute the inverse FFT of the product to get final convolved output
    Steps:
    1. Compute 2D FFT of input
    2. Compute 2D FFT of filter
    3. Element-wise multiply the two FFT results
    4. Compute inverse 2D FFT of the product to get convolved output
    @param in_width: Width of the input signal int
    @param in_height: Height of the input signal int
    @param filter_width: Width of the filter int
    @param filter_height: Height of the filter int
    @param input: Pointer to input signal (Complex numbers)
    @param filters: Pointer to filter signal (Complex numbers)
    @param output: Pointer to output signal (Float numbers)
*/
void cuda_operations::_2D_FFTConv(int in_width, int in_height, int filter_width, int filter_height,
                           cuComplex* input, cuComplex* filters, cuComplex* output) {
    //Implementation of 2D FFT Convolution kernel
    dim3 block(16, 16);
    dim3 grid((in_width + 15) / 16, (in_height + 15) / 16);

    // Steps:
    // 1. Compute 2D FFT of input
    utils::bitreversal<<<grid, block>>>(in_width, in_height, input);
    _1D_FFT<<<grid, block>>>(in_width, in_height, input, input);
    
    //get the transpose of the input for column-wise FFT
    utils::naivetranspose<<<grid, block>>>(in_width, in_height, input, input);

    // perform row-wise FFT again to complete 2D FFT
    utils::bitreversal<<<grid, block>>>(in_height, in_width, input);
    _1D_FFT<<<grid, block>>>(in_height, in_width, input, input);

    // 2. Compute 2D FFT of filter
    utils::bitreversal<<<grid, block>>>(filter_width, filter_height, filters);
    _1D_FFT<<<grid, block>>>(filter_width, filter_height, filters, filters);
    
    //get the transpose of the filter for column-wise FFT
    utils::naivetranspose<<<grid, block>>>(filter_width, filter_height, filters, filters);

    // perform row-wise FFT again to complete 2D FFT
    utils::bitreversal<<<grid, block>>>(filter_height, filter_width, filters);
    _1D_FFT<<<grid, block>>>(filter_height, filter_width, filters, filters);
    
    // 3. Element-wise multiply the two FFT results 
    int output_width = in_width; // for same conv - Basic FFTConv 
    int output_height = in_height;

    elementWiseMultiplyComplex<<<grid,block>>>(output_width, output_height, input, filters, output);

    // 4. Compute inverse 2D FFT of the product to get convolved output
    utils::bitreversal<<<grid, block>>>(output_width, output_height, output);
    _1D_IFFT<<<grid, block>>>(output_width, output_height, output, output);
    
    // get the transpose of the output for column-wise IFFT
    utils::naivetranspose<<<grid, block>>>(output_width, output_height, output, output);

    // perform row-wise IFFT again to complete 2D IFFT
    utils::bitreversal<<<grid, block>>>(output_height, output_width, output);
    _1D_IFFT<<<grid, block>>>(output_height, output_width, output, output);

}
