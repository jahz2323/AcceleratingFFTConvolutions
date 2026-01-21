#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <string>

/**
 * @file cuda_convolution.cu
 * @brief CUDA implementation of convolution operation 
**/

__global__ void _2DConv(int* in_width, int* in_height, int* in_channels,
                       int* filter_width, int* filter_height, int* out_channels,
                       int* stride, int* padding,
                       void* input, void* filters, void* output) {
    
    // cast input, filters, output to float pointers
    float* input_f = static_cast<float*>(input);
    float* filters_f = static_cast<float*>(filters);
    float* output_f = static_cast<float*>(output);

    // CUDA kernel for 2D convolution
    int local_id_x = threadIdx.x;
    int local_id_y = threadIdx.y;
    
    int out_x = blockIdx.x * blockDim.x + local_id_x;
    int out_y = blockIdx.y * blockDim.y + local_id_y;
    int out_c = blockIdx.z;

    // calculate output dims
    int out_width = (( *in_width - *filter_width + 2 * *padding) / *stride) + 1;
    int out_height = (( *in_height - *filter_height + 2 * *padding) / *stride) + 1;

    // bounds check
    if (out_x >= out_width || out_y >= out_height) {
        return;
    }
    float sum = 0.0f;

    // CHW loop
    for (int in_c = 0; in_c < *in_channels; in_c++) {
        for (int fh = 0; fh < *filter_height; fh++) {
            for (int fw = 0; fw < *filter_width; fw++) {
                int in_x = out_x * (*stride) + fw - (*padding);
                int in_y = out_y * (*stride) + fh - (*padding);
                if (in_x >= 0 && in_x < *in_width && in_y >= 0 && in_y < *in_height) {
                    int in_index = (in_c * (*in_width) * (*in_height)) + (in_x + in_y * (*in_width));
                    int filter_index = (out_c * (*in_channels) * (*filter_width) * (*filter_height)) +
                                       (in_c * (*filter_width) * (*filter_height)) +
                                       (fw + fh * (*filter_width));
                    sum += input_f[in_index] * filters_f[filter_index];
                }
            }
        }
    }

    float bias = 0.0f; // Assuming bias is zero for simplicity
    int out_index = (out_c * out_width * out_height) + (out_x + out_y * out_width);
    output_f[out_index] = sum + bias;
}
std::string addTwoStrings(const std::string& str1, const std::string& str2) {
    return str1 + str2;
}
int main(){ 
    // assume a 2D array size 28x28 with 1 channel
    // filter size 3x3, 1 channel, stride 1, padding 1
    int in_width = 28;
    int in_height = 28;
    int in_channels = 1;
    int filter_width = 3;
    int filter_height = 3;
    int out_channels = 1;
    int stride = 1;
    int padding = 1;
    int out_width = (( in_width - filter_width + 2 * padding) / stride) + 1;
    int out_height = (( in_height - filter_height + 2 * padding) / stride) + 1;

    // <<<Grid3D, Block3D>>>
    dim3 blockSize(16,16,1);
    int blocks_x = (in_width + blockSize.x - 1) / blockSize.x;
    int blocks_y = (in_height + blockSize.y - 1) / blockSize.y;
    dim3 gridSize(blocks_x, blocks_y, out_channels);

    size_t input_size = in_width * in_height * in_channels * sizeof(float);
    size_t filter_size = filter_width * filter_height * in_channels * out_channels * sizeof(float);
    size_t output_size = out_width * out_height * out_channels * sizeof(float);

    // initalize input, filter, output
    float* input = (float*)malloc(input_size);
    float* filter = (float*)malloc(filter_size);
    float* output = (float*)malloc(output_size);

    // Allocate device memory
    cudaMalloc((void**)&input, input_size);
    cudaMalloc((void**)&filter, filter_size);
    cudaMalloc((void**)&output, output_size);

    // copy input and filter to device
    cudaMemcpy(input, input, input_size, cudaMemcpyHostToDevice);
    cudaMemcpy(filter, filter, filter_size, cudaMemcpyHostToDevice);
    // launch kernel
    _2DConv<<<gridSize, blockSize>>>(&in_width, &in_height, &in_channels,
                                     &filter_width, &filter_height, &out_channels,
                                     &stride, &padding,
                                     input, filter, output);
    // free device memory
    cudaFree(input);
    cudaFree(filter);
    cudaFree(output);
    
    // free host memory
    free(input);
    free(filter);
    free(output);

    return 0; 
}