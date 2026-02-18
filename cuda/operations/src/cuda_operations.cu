#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <fstream>
#include <string>
#include <vector>


#include "cuda_operations.cuh"
#include "utils.cuh"

// For NVTX profiling
#include <nvtx3/nvToolsExt.h>

/**
    @brief 2D Pooling Kernel 
*/
template <cuda_operations::POOL_MODE POOL_MODE> 
__global__ void cuda_operations::_2DPool(int in_width, int in_height, int pool_width, int pool_height, int stride, int padding,  void* input, void* output){
    extern __shared__ float shared_pointer[]; 
    int pad_width = pool_width / 2; 
    int pad_height = pool_height / 2; 
    int block_idx = blockIdx.x * blockDim.x; 
    int block_idy = blockIdx.y * blockDim.y; 

    int global_offset = blockIdx.z * in_width*in_height; 
    int global_x_index; 
    int global_y_index; 

    //load global memory to shared memory with padding 
    for(int i = threadIdx.y; i < blockDim.y + 2 * pad_height; i = i + blockDim.y){
        for(int j = threadIdx.x; j < blockDim.x + 2 * pad_width; j = j + blockDim.x){
            global_x_index = block_idx + j - pad_width; 
            global_y_index = block_idy + i - pad_height; 
            if(global_x_index >= 0 && global_x_index < in_width && global_y_index >= 0 && global_y_index < in_height){
                shared_pointer[i * (blockDim.x + 2 * pad_width) + j] =(input)[global_offset + global_y_index * in_width + global_x_index];
            }
            else {
                shared_pointer[i * (blockDim.x + 2 * pad_width) + j] = 0.0f; // zero padding
            }
        }
    }

    __syncthreads();
    float pool_result;
    if(POOL_MODE == POOL_MODE::MAX_POOL){
        pool_result = -FLT_MAX; // initialize to smallest float
    }
    else if(POOL_MODE == POOL_MODE::AVERAGE_POOL){
        pool_result = 0.0f; // initialize to 0 for average pooling
    }

    // iterate over the pooling window in shared memory and compute the max or average
    for(int i = 0; i < pool_height; i++){
        for(int j = 0; j < pool_width; j++){
            float value = shared_pointer[(threadIdx.y + i) * (blockDim.x + 2 * pad_width) + threadIdx.x + j];
            if(POOL_MODE == POOL_MODE::MAX_POOL){
                pool_result = fmaxf(pool_result, value);
            }
            else if(POOL_MODE == POOL_MODE::AVERAGE_POOL){
                pool_result += value;
            }
        }
    }

    // for average pooling, divide the sum by the number of elements in the pooling window
    if(POOL_MODE == POOL_MODE::AVERAGE_POOL){
        pool_result /= (pool_width * pool_height);
    }

    // write the result back to global memory 
    (output)[global_offset + block_idy * (in_width / stride) + block_idx] = (pool_result);
}


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


template<bool isInverse, bool isRowWise>
__global__ void cuda_operations::OptimisedSharedMemory1DFFT(int width, cuComplex* input, cuComplex* output){
    /**
    Optimised Shared Memory 1D FFT/IFFT Kernel
    ---------------------------------------------
    32 - max threads per block in y dimension
    w - width of the input signal (assumed square for 2D FFT)
    Performs FFT or IFFT based on isInverse flag, 
    Ping-pong buffering in shared memory to avoid bank conflicts
    Row-wise or Column-wise FFT based on isRowWise flag
    In place FFT butterfly - each thread calculates its own butterfly operation
    ---------------------------------------------

    utilise shared_memory to sub divide the FFT computation 
    - 32 rows per block
    - for 2nd pass its 32 columns per block

    use inplace access - pingpong uses too much memory

    Steps: 
    1. Load data from global memory to shared memory (coalesced)
    2. Bit-reversal on shared memory
    3. Perform FFT/IFFT on shared memory
    4. Write back to global memory (transposed for column-wise FFT)

    **/
    extern __shared__ cuComplex shared_input_row[]; 
   
    int ty = threadIdx.y; // thread index in block row
    int tx = threadIdx.x; // thread index in block column

    int row = blockIdx.y * blockDim.y + ty; // Global row index

    if (tx >= width || row >= width)  return; // bounds check threadindex in x_dim should be less than width
   
    // Step 1. Read in global memory to shared memory - coalesced
    if (isRowWise){
        // row-wise FFT - read row into shared memory
        shared_input_row[tx] = input[row * width + tx];
    }
    else {
        // column-wise FFT - read column into shared memory
        shared_input_row[tx] = input[tx * width + row];
    }
    __syncthreads();

    /**
        32 - leading zeroes in (width) - 1 -> log2(width)
    */
    int stages = 32 -__clz(width) - 1; // log2(width)
    //if (row == 0 ) {printf("Stages: %d\n", stages);}

    // Step 2. Bit-reversal on shared memory
    /**
        __brev(tx) : returned the reversed bits of tx in 32 bit 
        need to shift right by (32 - stages) to get the correct index 
        for current block index 
    */
    unsigned int rev_n = __brev(tx) >> (32 - stages);
    if(rev_n > tx && tx < width){
        // swap
        cuComplex temp = shared_input_row[tx];
        shared_input_row[tx] = shared_input_row[rev_n];
        shared_input_row[rev_n] = temp;

    }
    __syncthreads();


    // Step 3. perform parallelised fft - each thread computes its own butterfly
    for (int s = 1; s<=stages; s++){
        /**
        Mapping threads to butterflies
        Each thread computes one butterfly operation 
        */
        int m  = 1 << s; // butterfly size
        int half_m = m >> 1; // distance between wings

        // Only first half of threads in the block are active for each stage as they compute the top wing of the butterfly, 
        // the bottom wing is computed by indexing with + half_m
        if ((tx < (width / 2))) { 
            int b_group = tx / half_m;  // which group of butterflies
            int b_j = tx % half_m;  // index within the butterfly
            int i = b_group * m + b_j;// top wing index
            int k = i + half_m; // bottom wing index


            // compute twiddle factor - 
            float angle = (isInverse ? -2.0f : 2.0f) * 3.141592653589793238462643383279502884f * b_j  / m;
            cuComplex w = make_cuComplex(cosf(angle), sinf(angle));

            // perform butterfly
            cuComplex u = shared_input_row[i]; 
            cuComplex t = cuCmulf(w, shared_input_row[k]);

            // write results back to shared memory
            shared_input_row[i] = cuCaddf(u, t); 
            shared_input_row[k] = cuCsubf(u, t);
        }
        __syncthreads();
    }

    // Step 4: transpose : write result to global[idy + idx*w] from shared memory
    // Not coalesced write, but necessary for column-wise FFT
    cuComplex fft_value = shared_input_row[tx]; // after final stage, the result is in shared_input_row due to ping-pong swapping
    if(isInverse){
        float scale = 1.0f / width;
        fft_value.x *= scale;
        fft_value.y *= scale;
    } 
    __syncthreads();
    // Transpose write
    if (isRowWise)
        output[row * width + tx] = fft_value; 
    else {
        // column-wise write
        output[tx * width + row] = fft_value;
    }
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
        float angle = -2.0f * 3.141592653589793238462643383279502884f * n * idx / in_width;
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
        float angle = 2.0f * 3.141592653589793238462643383279502884f  * idx * k / in_width; // check if macro is available 
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
        cuComplex wm = make_cuComplex(cosf(-2.0f * 3.141592653589793238462643383279502884f / m), sinf(-2.0f * 3.141592653589793238462643383279502884f / m));
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
        cuComplex wm = make_cuComplex(cosf(2.0f * 3.141592653589793238462643383279502884f / m), sinf(2.0f * 3.141592653589793238462643383279502884f / m));
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

    nvtxRangePushA("2D_FFTConv");
    //max thread dispatch per block 1024
    int maxThreadsPerBlock = 32; // 32x32 = 1024
    // dim3 block(fw, fh);
    // dim3 grid((w + block.x -1 ) / block.x, (h + block.y - 1) / block.y);
    dim3 block(maxThreadsPerBlock, maxThreadsPerBlock);
    dim3 grid((w + block.x -1 ) / block.x, (h + block.y - 1) / block.y);


    dim3 fft_block(1, maxThreadsPerBlock); // 1 thread per row
    dim3 fft_grid(1, (h + fft_block.y -1) / fft_block.y);
    cuComplex* d_temp;
    cudaMalloc(&d_temp, w * h * sizeof(cuComplex));

    // Steps:
    // 1. Compute 2D FFT of input
    nvtxRangePushA("InputFFT");
    cuda_operations::bitreversal<<<grid, block>>>(w, h, input);
    _1D_FFT<<<fft_grid, fft_block>>>(w, h, input, input);
    
    //get the transpose of the input for column-wise FFT
    cuda_operations::naivetranspose<<<grid, block>>>(w, h, input, d_temp);
    cudaMemcpy(input, d_temp, w * h * sizeof(cuComplex), cudaMemcpyDeviceToDevice);

    // perform row-wise FFT again to complete 2D FFT
    cuda_operations::bitreversal<<<grid, block>>>(h, w, input);
    _1D_FFT<<<fft_grid, fft_block>>>(h, w, input, input);
    nvtxRangePop(); // FFT_2D_FFTConv

    // 2. Compute 2D FFT of filter
    nvtxRangePushA("FilterFFT");
    cuda_operations::bitreversal<<<grid, block>>>(fw, fh, filters);
    _1D_FFT<<<fft_grid, fft_block>>>(fw, fh, filters, filters);
    
    //get the transpose of the filter for column-wise FFT
    cuda_operations::naivetranspose<<<grid, block>>>(fw, fh, filters, d_temp);
    cudaMemcpy(filters, d_temp, fw * fh * sizeof(cuComplex), cudaMemcpyDeviceToDevice);

    // perform row-wise FFT again to complete 2D FFT
    cuda_operations::bitreversal<<<grid, block>>>(fh, fw, filters);
    _1D_FFT<<<fft_grid, fft_block>>>(fh, fw, filters, filters);
    nvtxRangePop(); // FilterFFT
    
    // 3. Element-wise multiply the two FFT results 
    int output_width = w; // for same conv - Basic FFTConv 
    int output_height = h;

    nvtxRangePushA("ElementWiseMultiply_FFTConv");
    elementWiseMultiplyComplex<false><<<grid,block>>>(output_width, output_height, input, filters, output);
    nvtxRangePop(); // ElementWiseMultiply_FFTConv

    nvtxRangePushA("IFFT_2D_FFTConv");
    // 4. Compute inverse 2D FFT of the product to get convolved output
    cuda_operations::bitreversal<<<grid, block>>>(output_width, output_height, output);
    _1D_IFFT<<<fft_grid, fft_block>>>(output_width, output_height, output, output);
    
    // get the transpose of the output for column-wise IFFT
    cuda_operations::naivetranspose<<<grid, block>>>(output_width, output_height, output, d_temp);
    cudaMemcpy(output, d_temp, output_width * output_height * sizeof(cuComplex), cudaMemcpyDeviceToDevice);

    // perform row-wise IFFT again to complete 2D IFFT
    cuda_operations::bitreversal<<<grid, block>>>(output_height, output_width, output);
    _1D_IFFT<<<fft_grid, fft_block>>>(output_height, output_width, output, output);
    nvtxRangePop(); // IFFT_2D_FFTConv

    cudaFree(d_temp);
    nvtxRangePop();
}

#define MAX_SHARED_MEM 49152 // 48KB
void cuda_operations::Optimised2DFFTConv(int w, int h, cuComplex *input, cuComplex *filter, cuComplex *output){
    // Implementation of Optimised 2D FFT Convolution using Shared Memory FFT
    // Each block processes {n} rows
    // each thread processes 1 element in the row

    // convert w and h ints to long int 
    long int long_w = static_cast<long int>(w);
    long int long_h = static_cast<long int>(h);
     
    //PLAN:
    //FIRST PASS: ROW-WISE FFT
    //SECOND PASS: COLUMN-WISE FFT
    
    int rowsPerBlock = 16; // each block processes 16 rows
    dim3 block(w, rowsPerBlock);
    dim3 grid1(1, (w + rowsPerBlock -1) / rowsPerBlock);

    // Shared memory has to fit 1 block of data for the FFT computation - 16 rows of w elements each, complex numbers
    size_t sharedSize = block.x * block.y * sizeof(cuComplex); 
    
    // CHECK IF SIZE IS WITHIN LIMITS - hit limit for 
    if (sharedSize > MAX_SHARED_MEM){
        // std::cerr << "Error: Shared memory size exceeds limit! at:"<< sharedSize << std::endl;
        std::cout << "WARNING: Shared memory size exceeds limit! GO back to cuFFT implementation." << std::endl;
        return;
    } 

    //Scratch buffer for transpose
    cuComplex* d_temp;
    cudaMalloc(&d_temp, w * h * sizeof(cuComplex));

    // DEBUG Vector - print values before FFT
    std::vector<float> fft_output(w * h * 2); // real + imag
    cudaMemcpy(fft_output.data(), input, long_w * long_h * sizeof(cuComplex), cudaMemcpyDeviceToHost);
    std::cout << "Input before FFT: " << std::endl;
    utils::printConvResult(fft_output, w, h);

    //Pass1: Row-wise FFT
    OptimisedSharedMemory1DFFT<false, false><<<grid1, block, sharedSize>>>(long_w, input, d_temp);
    //Pass2: Column-wise FFT
    OptimisedSharedMemory1DFFT<false, false><<<grid1, block, sharedSize>>>(long_h, d_temp, input);

    //check output from d_temp and copy to vector float
    
    cudaMemcpy(fft_output.data(), input, w * h * sizeof(cuComplex), cudaMemcpyDeviceToHost);
    // Print FFT output for debugging
    std::cout << "FFT Output after 2D FFT: " << std::endl;
    utils::printConvResult(fft_output, w, h);
    
    // test inverse of input FFT to get back original input
    //Pass1: Row-wise IFFT
    OptimisedSharedMemory1DFFT<true, false><<<grid1, block, sharedSize>>>(long_w, input, d_temp);
    //Pass2: Column-wise IFFT
    OptimisedSharedMemory1DFFT<true, false><<<grid1, block, sharedSize>>>(long_h, d_temp, input);
    // Copy back to fft_output for checking
    cudaMemcpy(fft_output.data(), input, w * h * sizeof(cuComplex), cudaMemcpyDeviceToHost);
    std::cout << "Input after IFFT (should match original input): " << std::endl;
    utils::printConvResult(fft_output, w, h);

    // Repeat for filter
    OptimisedSharedMemory1DFFT<false, false><<<grid1, block, sharedSize>>>(long_w, filter, d_temp);
    OptimisedSharedMemory1DFFT<false, false><<<grid1, block, sharedSize>>>(long_h, d_temp, filter);
    
    // Element-wise multiplication
    dim3 block_mul(32, 32);
    dim3 grid_mul((w + block_mul.x -1 ) / block_mul.x, (h + block_mul.y - 1) / block_mul.y);
    elementWiseMultiplyComplex<false><<<grid_mul,block_mul>>>(w, h, input, filter, output);
    
    // Inverse FFT
    //Pass1: Row-wise IFFT
    OptimisedSharedMemory1DFFT<true, false><<<grid1, block, sharedSize>>>(long_w, output, d_temp);
    //Pass2: Column-wise IFFT
    OptimisedSharedMemory1DFFT<true, false><<<grid1, block, sharedSize>>>(long_h, d_temp, output);

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