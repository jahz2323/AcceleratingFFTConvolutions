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

__global__ void cuda_operations::_2DPool(
    int in_width, int in_height, 
    int pool_width, int pool_height, 
    int stride, int padding,  
    void* input, void* output,
    cuda_operations::POOL_MODE pool_mode
){
    // 2DPOOL with Average or Max pooling 
    // Using global memory for 
    //extern __shared__ float shared_pointer[]; 
    auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    auto idy = blockIdx.y * blockDim.y + threadIdx.y;

    int output_width = ((in_width - pool_width + 2 * padding) / stride) + 1;
    int output_height = ((in_height - pool_height + 2 * padding) / stride) + 1;

    if (idx >= output_width || idy >= output_height) return;
    if( pool_mode == POOL_MODE::MAX_POOL){
        float max_value = -FLT_MAX; // initialize to smallest float
        for(int i = 0; i < pool_width; i++){
                for (int j = 0; j < pool_height; j++){
                int in_x = idx * stride + i - padding;
                int in_y = idy * stride + j - padding;
                if(in_x >= 0 && in_x < in_width && in_y >= 0 && in_y < in_height){
                    float val = static_cast<float*>(input)[in_y * in_width + in_x];
                    if(val > max_value) max_value = val;
                }
            }
        }
        static_cast<float*>(output)[idy * output_width + idx] = max_value;
    }
    if(pool_mode == POOL_MODE::AVERAGE_POOL){
        float sum = 0.0f;
        int count = 0;
        for(int i = 0; i < pool_width; i++){
            for (int j = 0; j < pool_height; j++){
                int in_x = idx * stride + i - padding;
                int in_y = idy * stride + j - padding;
                if(in_x >= 0 && in_x < in_width && in_y >= 0 && in_y < in_height){
                    sum += static_cast<float*>(input)[in_y * in_width + in_x];
                    count++;
                }
            }
        }
        static_cast<float*>(output)[idy * output_width + idx] = count > 0 ? sum / count : 0.0f;
    }
}

// __global__ void cuda::operations::CutOffFrequency(int width, int height, cuComplex* input, cuComplex* output, float cutoff_freq){
//     // Cut off high frequency components in the frequency domain representation of the image
//     // Calculate the distance of each frequency component from the center of the frequency domain
//     // If the distance is greater than the cutoff frequency, set it to zero
//     int idx = blockIdx.x * blockDim.x + threadIdx.x;
//     int idy = blockIdx.y * blockDim.y + threadIdx.y;

//     if (idx >= width || idy >= height) return;

//     int center_x = width / 2;
//     int center_y = height / 2;

//     int freq_x = idx - center_x;
//     int freq_y = idy - center_y;

//     float distance = sqrtf(freq_x * freq_x + freq_y * freq_y);

//     if (distance > cutoff_freq) {
//         output[idy * width + idx] = make_cuComplex(0.0f, 0.0f);
//     } else {
//         output[idy * width + idx] = input[idy * width + idx];
//     }
// }

// void _2DSpectralPool(
//     int in_width, int in_height, 
//     int pool_width, int pool_height, 
//     int stride, int padding,  
//     cuComplex* input, cuComplex* output,
// ){
//         // Spectral Pooling implementation
//         // 1. Compute 2D FFT of input
//         // 2. Zero out high frequency components based on pool size (keep central low frequencies)
//         // 3. Compute inverse 2D FFT to get pooled output  
//     dim3 block(16,16);
//     dim3 grid((in_width + block.x -1 ) / block.x, (in_height + block.y - 1) / block.y);

//      // Step 1: Compute 2D FFT of input
//     int output_width = ((in_width - pool_width + 2 * padding) / stride) + 1;
//     int output_height = ((in_height - pool_height + 2 * padding) / stride) + 1;

//     cuda_operations::Forward2DFFT(in_width, in_height, input, output);
//     // Step 2: Zero out high frequency components
//     int freq_cutoff_x = output_width / 2;
//     int freq_cutoff_y = output_height / 2;
//     cuda_operations::CutOffFrequency<<<grid, block>>>(in_width, in_height, output, output, fminf(freq_cutoff_x, freq_cutoff_y));
//     // Step 3: Compute inverse 2D FFT to get pooled output
//     cuda_operations::Inverse2DFFT(in_width, in_height, output, output);

// }


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


__global__ void cuda_operations::overlap_add_kernel(
    const cuComplex* convolved_block,
    cuComplex* output,
    int block_w, int block_h,
    int start_x, int start_y,
    int out_w, int out_h
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx >= block_w || row >= block_h) return;

    int local_idx = row * block_w + idx;
    int output_x = start_x + idx;
    int output_y = start_y + row;

    if(output_x < out_w && output_y < out_h){
        int output_idx = output_y * out_w + output_x;
        float scale = 1 /  (block_w * block_h); // scale factor to prevent overflow, can be tuned based on expected value range of convolved_block
        //float scale = 1.0f; // no scaling for now, can be adjusted based on empirical testing
        cuComplex val = convolved_block[local_idx];
        // atomic add to handle overlapping regions
        atomicAdd(&output[output_idx].x, val.x * scale);
        atomicAdd(&output[output_idx].y, val.y * scale);
    }
}

__global__ void cuda_operations::tiling_and_extract_kernel(
    const cuComplex* input, 
    cuComplex* workspace_block, 
    int in_width, int in_height,
    int start_x, int start_y,
    int block_w, int block_h
){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx >= block_w || idy >= block_h) return;

    int local_idx = idy * block_w + idx;
    int img_x = start_x + idx; // where we are relative to the image for current block
    int img_y = start_y + idy; 

    if(img_x < in_width && img_y < in_height){
        workspace_block[local_idx] = input[img_y * in_width + img_x];
    } else {
        workspace_block[local_idx] = make_cuComplex(0.0f, 0.0f); // zero pad
    }
}

void cuda_operations::FFT_OVA_Conv(
    int in_width, int in_height, 
    int filter_width, int filter_height, 
    int stride, int padding,
    int num_streams,
    cudaStream_t streams[],
    cuComplex* d_input_complex, cuComplex* d_filter_complex, 
    cuComplex* d_output_complex, cuComplex* workspace_block,
    cuComplex* d_workspaces[], cuComplex* d_scratches[],
    int segment_w, int segment_h,
    int block_w, int block_h,
    int num_blocks_w, int num_blocks_h,
    int total_blocks
){
    /**
            Optimised Overlap-Add Convolution using FFT
            Data d with dim (N,N) filter f with dim (K,K)
            1. Segment data D into non-overlapping blocks of kxk 
            2. Zero pad to k+k-1 
            3. Compute FFT of filter
            4. for each block do : 
            - zero pad to k+k-1
            - compute FFT of block
            - element-wise multiply in frequency domain
            - compute inverse FFT to get convolved block
            5. Overlap-add the convolved blocks to get final output
        */
    // Input is contigous original image, 
    // filter is original filter
    // workspace_block zero padded to power of two (target - k + 1) 
    // perform a sliding window batch over image , cache filter and perform 2dfft on each block, then overlap add to output
    int in_w = in_width;
    int in_h = in_height;
    int f_w = filter_width;
    int f_h = filter_height;
    int out_w = ((in_w - f_w + 2 * padding)/stride + 1);
    int out_h = ((in_h - f_h + 2 * padding)/stride + 1);
    std::cout << "Output dimensions: " << out_w << " x " << out_h << std::endl;
    dim3 block_size(16,16); 
    dim3 block_grid((block_w + block_size.x - 1) / block_size.x, 
                (block_h + block_size.y - 1) / block_size.y);
    
    std::cout << "Total blocks to process: " << total_blocks << std::endl;
    std::cout << "Params entry: in_w: " << in_w << " in_h: " << in_h << " f_w: " << f_w << " f_h: " << f_h << std::endl;
    std::cout << "block_w: " << block_w << " block_h: " << block_h << " segment_w: " << segment_w << " segment_h: " << segment_h << std::endl;
    std::cout << "num_blocks_w: " << num_blocks_w << " num_blocks_h: " << num_blocks_h << std::endl;
    nvtxRangePushA("FFT_OVA_Conv");

    //check if d_fitler_complex has value 
    std::vector<cuComplex> h_filter_complex(block_w * block_h);
    cudaMemcpy(h_filter_complex.data(), d_filter_complex, block_w * block_h * sizeof(cuComplex), cudaMemcpyDeviceToHost);
    utils::checkcuComplexArray(h_filter_complex.data(), block_w, block_h, "OVA_FILTER_PRIOR_FFT");
 
    // Compute FFT of filter once and reuse for all blocks
    cuda_operations::Forward2DFFT(block_w, block_h, d_filter_complex, d_output_complex); // in-place FFT of filter
    cudaDeviceSynchronize();

    // DEBUG: copy back filter FFT to host and print sum of real parts as a simple check
    cudaMemcpy(h_filter_complex.data(), d_filter_complex, block_w * block_h * sizeof(cuComplex), cudaMemcpyDeviceToHost);
    utils::checkcuComplexArray(h_filter_complex.data(), block_w, block_h, "OVA_FILTER_AFTER_FFT");
 
     // ENTIRE SNIPPET CAN BE MANAGED IN SETUP GPU
    // create a scratch buffer of d_output_complex size 
    // cuComplex* d_scratch;
    // cudaMalloc(&d_scratch, block_w * block_h * sizeof(cuComplex));
    // cudaMemset(d_scratch, 0, block_w * block_h * sizeof(cuComplex));


    int num_elements = block_w * block_h;
    int threads_per_block = 64;
    int num_blocks = (num_elements + threads_per_block - 1) / threads_per_block;
    
    //Need a workspace and scratch space per stream 
    for (int block_idx = 0; block_idx < total_blocks; ++block_idx){
        int stream_idx = block_idx % num_streams;

        int block_x = block_idx % num_blocks_w;
        int block_y = block_idx / num_blocks_w;

        int start_x = block_x * segment_w;
        int start_y = block_y * segment_h;

        // extract block to workspace with zero padding
        cuda_operations::tiling_and_extract_kernel<<<block_grid, block_size>>>(
            d_input_complex, d_workspaces[stream_idx], 
            in_w, in_w, 
            start_x, 
            start_y, 
            block_w, block_h
        ); 
       
        // compute FFT of block in-place in workspace
        cuda_operations::Forward2DFFT(block_w, block_h, d_workspaces[stream_idx], d_scratches[stream_idx]);
        // element-wise multiply in frequency domain with filter
        cuda_operations::elementWiseMultiplyComplex<false><<<num_blocks, threads_per_block>>>(
            block_w,block_h , d_workspaces[stream_idx], d_filter_complex, d_workspaces[stream_idx]
        );  
         // compute inverse FFT to get convolved block in workspace
        cuda_operations::Inverse2DFFT(block_w, block_h, d_workspaces[stream_idx], d_scratches[stream_idx]);
        
        // store the y block to global memory
        cuda_operations::overlap_add_kernel<<<block_grid, block_size>>>(
            d_workspaces[stream_idx], d_output_complex,
            block_w, block_h, 
            start_x, 
            start_y,
            out_w, out_h
        );
    }

    //SYNC STREAMS
    for (int i = 0; i < num_streams; i++) {
        cudaStreamSynchronize(streams[i]);
    }

    nvtxRangePop(); // FFT_OVA_Conv
}

void cuda_operations::Forward2DFFT( int in_width, int in_height, cuComplex* input, cuComplex* output){
    dim3 bitrev_block(32, 32);
    dim3 bitrev_grid((in_width + bitrev_block.x -1 ) / bitrev_block.x, (in_height + bitrev_block.y - 1) / bitrev_block.y);


    dim3 fft_block(1, 32); // 1 thread per row
    dim3 fft_grid(1, (in_height + fft_block.y -1) / fft_block.y);

    cuda_operations::bitreversal<<<bitrev_grid, bitrev_block>>>(in_width, in_height, input);
    // Perform row-wise FFT
    cuda_operations::_1D_FFT<<<fft_grid, fft_block>>>(in_width, in_height, input, input);

    // Transpose the result for column-wise FFT
    cuda_operations::naivetranspose<<<bitrev_grid, bitrev_block>>>(in_width, in_height, input, output);
    cudaMemcpy(input, output, in_width * in_height * sizeof(cuComplex), cudaMemcpyDeviceToDevice);

    cuda_operations::bitreversal<<<bitrev_grid, bitrev_block>>>(in_width, in_height, input);
    // Perform column-wise FFT
    cuda_operations::_1D_FFT<<<fft_grid, fft_block>>>(in_height, in_width, input, input);
}

void cuda_operations::Inverse2DFFT(int in_width, int in_height, cuComplex* input, cuComplex* output){
    dim3 bitrev_block(32, 32);
    dim3 bitrev_grid((in_width + bitrev_block.x -1 ) / bitrev_block.x, (in_height + bitrev_block.y - 1) / bitrev_block.y);


    dim3 fft_block(1, 32); // 1 thread per row
    dim3 fft_grid(1, (in_height + fft_block.y -1) / fft_block.y);
    
    cuda_operations::bitreversal<<<bitrev_grid, bitrev_block>>>(in_width, in_height, input);
    // Perform row-wise IFFT
    cuda_operations::_1D_IFFT<<<fft_grid, fft_block>>>(in_width, in_height, input, input);

    // Transpose the result for column-wise IFFT
    cuda_operations::naivetranspose<<<bitrev_grid, bitrev_block>>>(in_width, in_height, input, output);
    cudaMemcpy(input, output, in_width * in_height * sizeof(cuComplex), cudaMemcpyDeviceToDevice);

    cuda_operations::bitreversal<<<bitrev_grid, bitrev_block>>>(in_width, in_height, input);
    // Perform column-wise IFFT
    cuda_operations::_1D_IFFT<<<fft_grid, fft_block>>>(in_height, in_width, input, input);
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

    //max thread dispatch per block 1024
    int maxThreadsPerBlock = 32; // 32x32 = 1024

    dim3 block(maxThreadsPerBlock, maxThreadsPerBlock);
    dim3 grid((w + block.x -1 ) / block.x, (h + block.y - 1) / block.y);

    nvtxRangePushA("2D_FFTConv");
    // Steps:
    // 1. Compute 2D FFT of input
    nvtxRangePushA("InputFFT");
    cuda_operations::Forward2DFFT(w, h, input, output);
    nvtxRangePop(); // InputFFT
    // 2. Compute 2D FFT of filter
    nvtxRangePushA("FilterFFT");
    cuda_operations::Forward2DFFT(w, h, filters, output);
    nvtxRangePop(); // FilterFFT
    // 3. Element-wise multiply the two FFT results 
    int output_width = w; // for same conv - Basic FFTConv 
    int output_height = h;
    nvtxRangePushA("ElementWiseMultiply_FFTConv");
    elementWiseMultiplyComplex<false><<<grid,block>>>(output_width, output_height, input, filters, output);
    nvtxRangePop(); // ElementWiseMultiply_FFTConv
    nvtxRangePushA("IFFT_2D_FFTConv");
    // 4. Compute inverse 2D FFT of the product to get convolved output
    cuda_operations::Inverse2DFFT(w, h, output, input);
    nvtxRangePop(); // IFFT_2D_FFTConv

    nvtxRangePop(); // 2D_FFTConv
    // cudaFree(d_temp);
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