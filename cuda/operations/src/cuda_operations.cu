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
    @NOTE: 
    had an issue with mapping indexes from 1D to 2D in the kernel, 
    Change to a cross-correlation operation [n+k] instead of convolution [n-k] to fix it.
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
            int n_start_pos = idy * stride + k - padding;
            int m_start_pos = idx * stride + l - padding;
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


template<bool isInverse>
__global__ void cuda_operations::OptimisedSharedMemory1DFFT(int width, cuComplex* input, cuComplex* output){
    /**
    Steps: 
    1. shared_input_memory[BLOCK_SIZE][BLOCK_SIZE]
    2. Coalesced rowdata from global memory to shared_input_memory[threadIdx.y][threadIdx.x]
    3. Perform FFT on shared memory data
    4. transpose : write result to global[idy + idx*w] from shared memory
    5. sync - barrier 
    // launch second pass for column-wise FFT
    **/
    extern __shared__ cuComplex shared_input_row[]; 
   
    int ty = threadIdx.y; // row in block 
    int tx = threadIdx.x; // col in block
    int idx = blockIdx.x * blockDim.x + tx; // global col index
    int idy = blockIdx.y * blockDim.y + ty; // global row index
    // bounds check
    if(idx >= width || idy >= width) return;

    // Allocate ping-pong buffers in shared memory
    cuComplex *ping = &shared_input_row[ty   * width];
    cuComplex *pong = &shared_input_row[(ty + blockDim.y) * width];

    
    // Step 2: Coalesced rowdata from global memory to shared_input_memory[threadIdx.y][threadIdx.x]
    shared_input_row[ty * width + tx] = input[idy * width + idx];
    __syncthreads();

    int stages = 31 - __clz(width); // log2(width)
    
    // Step 3: Bit-reversal on shared memory
    unsigned int rev_n = __brev(tx) >> (32 - stages); // log2(width)
    if(rev_n > tx){
        // swap
        cuComplex temp = shared_input_row[ty * width + tx];
        shared_input_row[ty * width + tx] = shared_input_row[ty * width + rev_n];
        shared_input_row[ty * width + rev_n] = temp;
    }
    __syncthreads();

    // perform parallelised fft - each thread computes its own butterfly
    for (int s = 1; s<=stages; s++){
        int m  = 1 << s; // butterfly size
        int half_m = m >> 1; // distance between wings
        
        /**
        Mapping threads to butterflies
        Each thread computes one butterfly operation 
        */
        int section = tx / half_m; // which section of butterflies
        int group = tx / m;  // 
        int j = tx % half_m; 

        if (section % 2 == 0){
            int i = group * m + j;
            int k   = i + half_m;
            // compute twiddle factor - 
            float angle = (isInverse ? 2.0f : -2.0f) * 3.14159265359f * j / m;
            cuComplex w = make_cuComplex(cosf(angle), sinf(angle));

            // perform butterfly
            cuComplex u = shared_input_row[ty * width + i];
            cuComplex t = cuCmulf(w, shared_input_row[ty * width + k]);

            shared_input_row[ty * width + i] = cuCaddf(u, t);
            shared_input_row[ty * width + k] = cuCsubf(u, t);
        }
        __syncthreads();
    }

    // Step 4: transpose : write result to global[idy + idx*w] from shared memory
    // Not coalesced write, but necessary for column-wise FFT
    cuComplex fft_value = shared_input_row[ty * width + tx];
    if(isInverse){
        float scale = 1.0f / width;
        fft_value.x *= scale;
        fft_value.y *= scale;
    } 
    output[idx * width + idy] = fft_value; // transposed write
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

   float scale = 1.0f / width;
    for(int i = 0; i < width; i++){
        row_data[i] = make_cuComplex(row_data[i].x * scale, row_data[i].y * scale);
        //DEBUG:
        // row_data[i].x = 1.0f; 
        // row_data[i].y = 1.0f;
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
    @param w: Width of the input signal int
    @param h: Height of the input signal int
    @param fw: Width of the filter int
    @param fh: Height of the filter int
    @param input: Pointer to input signal (Complex numbers)
    @param filters: Pointer to filter signal (Complex numbers)
    @param output: Pointer to output signal (Float numbers)
*/
void cuda_operations::_2D_FFTConv(int w, int h, int fw, int fh,
                           cuComplex* input, cuComplex* filters, cuComplex* output) {
    //Implementation of 2D FFT Convolution kernel
    //dim3 block(w + fw -1 , h + fh -1);
    
    //max thread dispatch per block 1024
    int maxThreadsPerBlock = 32; // 32x32 = 1024
    // dim3 block(fw, fh);
    // dim3 grid((w + block.x -1 ) / block.x, (h + block.y - 1) / block.y);
    dim3 block(maxThreadsPerBlock, maxThreadsPerBlock);
    dim3 grid((w + block.x -1 ) / block.x, (h + block.y - 1) / block.y);

    cuComplex* d_temp;
    cudaMalloc(&d_temp, w * h * sizeof(cuComplex));

    // Steps:
    // 1. Compute 2D FFT of input
    cuda_operations::bitreversal<<<grid, block>>>(w, h, input);
    _1D_FFT<<<grid, block>>>(w, h, input, input);
    
    //get the transpose of the input for column-wise FFT
    cuda_operations::naivetranspose<<<grid, block>>>(w, h, input, d_temp);
    cudaMemcpy(input, d_temp, w * h * sizeof(cuComplex), cudaMemcpyDeviceToDevice);

    // perform row-wise FFT again to complete 2D FFT
    cuda_operations::bitreversal<<<grid, block>>>(h, w, input);
    _1D_FFT<<<grid, block>>>(h, w, input, input);

    // 2. Compute 2D FFT of filter
    cuda_operations::bitreversal<<<grid, block>>>(fw, fh, filters);
    _1D_FFT<<<grid, block>>>(fw, fh, filters, filters);
    
    //get the transpose of the filter for column-wise FFT
    cuda_operations::naivetranspose<<<grid, block>>>(fw, fh, filters, d_temp);
    cudaMemcpy(filters, d_temp, fw * fh * sizeof(cuComplex), cudaMemcpyDeviceToDevice);

    // perform row-wise FFT again to complete 2D FFT
    cuda_operations::bitreversal<<<grid, block>>>(fh, fw, filters);
    _1D_FFT<<<grid, block>>>(fh, fw, filters, filters);
    
    // 3. Element-wise multiply the two FFT results 
    int output_width = w; // for same conv - Basic FFTConv 
    int output_height = h;

    elementWiseMultiplyComplex<false><<<grid,block>>>(output_width, output_height, input, filters, output);

    // 4. Compute inverse 2D FFT of the product to get convolved output
    cuda_operations::bitreversal<<<grid, block>>>(output_width, output_height, output);
    _1D_IFFT<<<grid, block>>>(output_width, output_height, output, output);
    
    // get the transpose of the output for column-wise IFFT
    cuda_operations::naivetranspose<<<grid, block>>>(output_width, output_height, output, d_temp);
    cudaMemcpy(output, d_temp, output_width * output_height * sizeof(cuComplex), cudaMemcpyDeviceToDevice);

    // perform row-wise IFFT again to complete 2D IFFT
    cuda_operations::bitreversal<<<grid, block>>>(output_height, output_width, output);
    _1D_IFFT<<<grid, block>>>(output_height, output_width, output, output);

    cudaFree(d_temp);
}

#define BLOCK_SIZE 1024
void cuda_operations::Optimised2DFFTConv(int w, int h, cuComplex *input, cuComplex *filter, cuComplex *output){
    // Implementation of Optimised 2D FFT Convolution using Shared Memory FFT
    // One block handles 8 rows , width must be <= BLOCK_SIZE

    if(w > BLOCK_SIZE){
        std::cerr << "Error: Width:" << w << " exceeds BLOCK_SIZE:" << BLOCK_SIZE << " for Optimised2DFFTConv" << std::endl;
        return;
    }
    //FIRST PASS: ROW-WISE FFT
    dim3 block(w, 8); // 1024 threads per block
    dim3 grid1(1, (h + block.y -1) / block.y);
    //SECOND PASS: COLUMN-WISE FFT
    dim3 grid2((h + block.y -1) / block.y, 1);
    size_t sharedSize = block.x * block.y * sizeof(cuComplex);

    //Scratch buffer for transpose
    cuComplex* d_temp;
    cudaMalloc(&d_temp, w * h * sizeof(cuComplex));

    //Pass1: Row-wise FFT
    OptimisedSharedMemory1DFFT<false><<<grid1, block, sharedSize>>>(w, input, d_temp);
    //Pass2: Column-wise FFT
    OptimisedSharedMemory1DFFT<false><<<grid2, block, sharedSize>>>(h, d_temp, input);

    // Repeat for filter
    OptimisedSharedMemory1DFFT<false><<<grid1, block, sharedSize>>>(w, filter, d_temp);
    OptimisedSharedMemory1DFFT<false><<<grid2, block, sharedSize>>>(h, d_temp, filter);

    // Element-wise multiplication
    dim3 block_mul(32, 32);
    dim3 grid_mul((w + block_mul.x -1 ) / block_mul.x, (h + block_mul.y - 1) / block_mul.y);
    elementWiseMultiplyComplex<true><<<grid_mul,block>>>(w, h, input, filter, output);

    // Inverse FFT
    //Pass1: Row-wise IFFT
    OptimisedSharedMemory1DFFT<true><<<grid1, block, sharedSize>>>(w, output, d_temp);
    //Pass2: Column-wise IFFT
    OptimisedSharedMemory1DFFT<true><<<grid2, block, sharedSize>>>(h, d_temp, output);

    cudaFree(d_temp);
}


/**
    @brief 2D cuFFTConv CUDA kernel
    Take input and filter as Complex* device pointers 
    take input dimensions and filter dimensions
    output the convolved result to output pointers as Complex*,
    Once in the frequency domain, perform element-wise multiplication
    then compute the inverse FFT of the product to get final convolved output
*/
void cuda_operations::_2DcuFFTConv(cufftHandle plan, int in_width, int in_height, int filter_width, int filter_height,
                           cuComplex* input, cuComplex* filters, cuComplex* output) {
    // Wrapper for cuFFT-based 2D convolution
    // Steps:
    // 1. Compute 2D FFT of input using cuFFT
    // 2. Compute 2D FFT of filter using cuFFT
    // 3. Element-wise multiply the two FFT results
    // 4. Compute inverse 2D FFT of the product using cuFFT to get convolved output
   

    // Create 2D FFT plan
    if (cufftPlan2d(&plan, in_height, in_width, CUFFT_C2C) != CUFFT_SUCCESS) {
        std::cerr << "CUFFT Error: Unable to create plan" << std::endl;
        return;
    }
    // 1. Compute 2D FFT of input
    if (cufftExecC2C(plan, input, input, CUFFT_FORWARD) != CUFFT_SUCCESS) {
        std::cerr << "CUFFT Error: Unable to execute plan for input" << std::endl;
        cufftDestroy(plan);
        return;
    }
    // 2. Compute 2D FFT of filter
    if (cufftExecC2C(plan, filters, filters, CUFFT_FORWARD) != CUFFT_SUCCESS) {
        std::cerr << "CUFFT Error: Unable to execute plan for filters" << std::endl;
        cufftDestroy(plan);
        return;
    }
    // 3. Element-wise multiply the two FFT results
    int output_width = in_width; // for same conv - Basic FFTConv
    int output_height = in_height;
    int maxThreadsPerBlock = 32; // 32x32 = 1024
    dim3 block(maxThreadsPerBlock, maxThreadsPerBlock);
    dim3 grid((output_width + block.x -1 ) / block.x, (output_height + block.y - 1) / block.y);
    elementWiseMultiplyComplex<false><<<grid,block>>>(output_width, output_height, input, filters, output);

    cudaDeviceSynchronize();

    // 4. Compute inverse 2D FFT of the product to get convolved output
    if (cufftExecC2C(plan, output, output, CUFFT_INVERSE) != CUFFT_SUCCESS) {
        std::cerr << "CUFFT Error: Unable to execute inverse plan for output" << std::endl;
        cufftDestroy(plan);
        return;
    }
    // Normalize the output
    int total_size = output_width * output_height;
    float scale = 1.0f / total_size;
   
    // Kernel to scale the output
    cuda_operations::scaleOutput<<<grid, block>>>(output_width, output_height, output, scale);

}

/**
    @brief Element-wise multiplication of complex numbers CUDA kernel
    @param width: Width of the input signal int
    @param height: Height of the input signal int
    @param input: Pointer to input signal (complex numbers)
    @param filters: Pointer to filter signal (complex numbers)
    @param output: Pointer to output signal (complex numbers)
*/
template<bool IsStandard>
__global__ void cuda_operations::elementWiseMultiplyComplex(int width, int height, cuComplex* input, cuComplex* filters, cuComplex* output){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if(idx >= width || idy >= height) return;
    if (IsStandard){
        // Standard element-wise multiplication
        int index = idy * width + idx;
        cuComplex input_val = (input)[index];
        cuComplex filter_val = (filters)[index];
        cuComplex product = cuCmulf(input_val, filter_val);
        (output)[index] = product;
        
    }
    else {
        // Conjugate element-wise multiplication
        int index = idy * width + idx;
        cuComplex input_val = (input)[index];
        cuComplex filter_val = cuConjf((filters)[index]);
        cuComplex product = cuCmulf(input_val, filter_val);
        (output)[index] = product;
    }
}


/**
    @author https://github.com/dbids-EC527/fft/blob/master/base_code/fft_2d.c
*/
__global__ void cuda_operations::bitreversal(int n, void* storage){
    int i, j;
    for (i=1, j=0; i < n; i++){
        int bit = n >>1; 
        for(; j & bit; bit>>=1){
            j ^= bit;
        }
        j ^= bit;
        cuComplex temp;
        if(i < j){
            // swap
            temp = static_cast<cuComplex*>(storage)[i];
            static_cast<cuComplex*>(storage)[i] = static_cast<cuComplex*>(storage)[j];
            static_cast<cuComplex*>(storage)[j] = temp;
        }
    } 
}

__global__ void cuda_operations::bitreversal(int width, int height, cuComplex* data){
    auto idx  = blockIdx.x * blockDim.x + threadIdx.x;
    auto idy  = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < width && idy < height){
        unsigned int rev_n = 0; 
        int temp_x = idx;
        int bits = static_cast<int>(log2f((float)width));
        for(int i = 0; i < bits; i++){
            rev_n = (rev_n << 1) | (temp_x & 1);
            temp_x >>= 1;
        }
        if  (rev_n > idx){
            // swap
            cuComplex temp = data[idy * width + idx];
            data[idy * width + idx] = data[idy * width + rev_n];
            data[idy * width + rev_n] = temp;
        }
    }
}

__global__ void cuda_operations::float2complex(int width, int height, float* input, cuComplex* output){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if(idx >= width || idy >= height) return;

    int index = idy * width + idx;
    float real_value = static_cast<float*>(input)[index];
    (output)[index] = make_cuComplex(real_value, 0.0f);
}

__global__ void cuda_operations::complex2float(int width, int height, cuComplex* input, float* output){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if(idx >= width || idy >= height) return;

    int index = idy * width + idx;
    //float scale = 1.0f / (width * height);
    cuComplex cvalue = (input)[index];
    // static_cast<float*>(output)[index] = cuCrealf(cvalue) * scale;
    static_cast<float*>(output)[index] = cuCrealf(cvalue);
}

__global__ void cuda_operations::copy(float* odata, float* idata){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int width = gridDim.x * blockDim.x;
    int index = idy * width + idx;
    odata[index] = idata[index];
}

__global__ void cuda_operations::copyComplex(cuComplex* odata, cuComplex* idata){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int width = gridDim.x * blockDim.x;
    int index = idy * width + idx;
    odata[index] = idata[index];
}

__global__ void cuda_operations::naivetranspose(int width, int height, cuComplex* input, cuComplex* output){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if(idx >= width || idy >= height) return;

    int in_index = idy * width + idx;
    int out_index = idx * height + idy;
    output[out_index] = input[in_index];
}

__global__ void cuda_operations::scaleOutput(int width, int height, cuComplex* output, float scale){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if(idx >= width || idy >= height) return;

    int index = idy * width + idx;
    cuComplex cvalue = output[index];
    output[index] = make_cuComplex(cvalue.x * scale, cvalue.y * scale);
}