#include "convolution_test.cuh"
/**
    @brief Test function for 1D Convolution

**/
void convolution_test::test1DConvolution(){
    std ::cout << "Running 1D Convolution Test..." << std::endl;
    //define static params
    int in_width = 5; 
    int filter_width = 3;
    int stride = 1;
    int padding = 0;
    int out_width = ((in_width - filter_width + 2 * padding) / stride) + 1;

    //allocate and initialize host memory
    std::vector<float> h_input = {1, 2, 3, 4, 5};
    std::vector<float> h_filter = {1, 0, 1};
    std::vector<float> h_output(out_width, 0.0f);

    //allocate device memory
    float *d_input, *d_filter, *d_output;
    cudaMalloc((void**)&d_input, in_width * sizeof(float));
    cudaMalloc((void**)&d_filter, filter_width * sizeof(float));
    cudaMalloc((void**)&d_output, out_width * sizeof(float));

    //copy data to device
    cudaMemcpy(d_input, h_input.data(), in_width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_filter.data(), filter_width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, h_output.data(), out_width * sizeof(float), cudaMemcpyHostToDevice);

    //launch kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (out_width + threadsPerBlock - 1) / threadsPerBlock;
    cuda_operations::_1DConv<<<blocksPerGrid, threadsPerBlock>>>(
        in_width, filter_width, stride, padding,
        d_input, d_filter, d_output
    );
    cudaDeviceSynchronize();

    //copy result back to host
    cudaMemcpy(h_output.data(), d_output, out_width * sizeof(float), cudaMemcpyDeviceToHost);
    
    //free device memory
    cudaFree(d_input);
    cudaFree(d_filter);
    cudaFree(d_output);
    //print output
    std::cout << "Convolution Output: ";
    for (const auto& val : h_output) {
        std::cout << val << " ";
    }
    std::cout << std::endl;

    std::cout << "1D Convolution test executed." << std::endl;
}


/**
    @brief Test function for 2D Convolution
    Custom 2dconv kernel
    torch 2conv
    custom 2dfftconv
**/
void convolution_test::test2DConvolution(int test_input_height,
                                            int test_input_width,
                                            int test_filter_height,
                                            int test_filter_width,
                                            int test_stride,
                                            int test_padding
    ){
    std ::cout << "Running 2D Convolution Test..." << std::endl;

    // Static clock
    cudaEvent_t start, stop;
    float milliseconds = 0;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Implementation for 2D convolution test would go here
    int in_width = test_input_width;
    int in_height = test_input_height;
    int filter_width = test_filter_width;
    int filter_height = test_filter_height;
    int stride = test_stride;
    int padding = test_padding;
    int out_width = ((in_width - filter_width + 2 * padding) / stride) + 1;
    int out_height = ((in_height - filter_height + 2 * padding) / stride) + 1;

    //allocate and initialize host memory
    // std::vector<float> h_input = {
    //     1, 2, 3, 4,
    //     5, 6, 7, 8,
    //     9,10,11,12,
    //    13,14,15,16
    // };
    // std::vector<float> h_filter = {
    //     1,0,-1,
    //     1,0,-1,
    //     1,0,-1
    // };

    // generate random input and filter
    std::vector<float> h_input(in_width * in_height);
    std::vector<float> h_filter(filter_width * filter_height);
    for (int i = 0; i < in_width * in_height; ++i) {
        h_input[i] = static_cast<float>(rand() % 10);
    }
    for (int i = 0; i < filter_width * filter_height; ++i) {
        h_filter[i] = static_cast<float>(rand() % 3 - 1); // values between -1 and 1
    }   
    std::vector<float> h_output(out_width * out_height, 0.0f);

    // Create clone of input, filter and out for Spectral method 
    // check for pow2 
    int size_of_input = in_width * in_height;
    int size_of_filter = filter_width * filter_height;
    
    //Target dim size for FFTConv N + K - 1 after power of two
    int target_dimensions_height = in_height + filter_height - 1;
    int target_dimensions_width = in_width + filter_width - 1;
    
    int fft_h = utils::nextPowerOfTwo(target_dimensions_height);
    int fft_w = utils::nextPowerOfTwo(target_dimensions_width);

    //set torch device 
    torch::Device device(torch::kCUDA, 0);
    
    // create torch tensors 
    auto gpu_options = torch::TensorOptions().dtype(torch::kFloat).device(device);
    auto cpu_options = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCPU);

    // Create input, filter and output tensors
    torch::Tensor input_tensor = torch::from_blob(h_input.data(), {1, 1, in_height, in_width}, cpu_options).to(device);
    torch::Tensor filter_tensor = torch::from_blob(h_filter.data(), {1, 1, filter_height, filter_width}, cpu_options).to(device);
    torch::Tensor output_tensor = torch::zeros({1, 1, out_height, out_width}, gpu_options);

    // calc input_pad dims 
    int padded_in_width =  fft_w - in_width;
    int padded_in_height = fft_h - in_height;

    // calc filter_pad dims
    int padded_filter_width =  fft_w - filter_width;
    int padded_filter_height = fft_h - filter_height;
    
    // pad to target dimensions 
    torch::Tensor padded_input = torch::constant_pad_nd(input_tensor, {0, padded_in_width, 0, padded_in_height}, 0);
    torch::Tensor padded_filter = torch::constant_pad_nd(filter_tensor, {0, padded_filter_width, 0, padded_filter_height}, 0);
    torch::Tensor padded_output = torch::zeros({fft_h, fft_w}, gpu_options);

    // Bring back to host as float*
    float *padded_input_ptr, *padded_filter_ptr, *padded_output_ptr;
    padded_input_ptr = padded_input.data_ptr<float>();
    padded_filter_ptr = padded_filter.data_ptr<float>();
    padded_output_ptr = padded_output.data_ptr<float>();

    // allocate space for complex versions
    cuComplex *d_padded_input, *d_padded_filter, *d_padded_output;
    cudaMalloc((void**)&d_padded_input, fft_h * fft_w * sizeof(cuComplex));
    cudaMalloc((void**)&d_padded_filter, fft_h * fft_w * sizeof(cuComplex));
    cudaMalloc((void**)&d_padded_output, fft_h * fft_w * sizeof(cuComplex));

    //Copy float* to complex* - pass to _2D_FFTConv
    dim3 fft_threadsPerBlock(16, 16);
    dim3 fft_blocksPerGrid((fft_w + fft_threadsPerBlock.x - 1) / fft_threadsPerBlock.x,
                           (fft_h + fft_threadsPerBlock.y - 1) / fft_threadsPerBlock.y);
    utils::float2complex<<<fft_blocksPerGrid, fft_threadsPerBlock>>>(
        fft_w, fft_h, padded_input_ptr, d_padded_input
    );
    dim3 filter_threadsPerBlock(16, 16);
    dim3 filter_blocksPerGrid((fft_w + filter_threadsPerBlock.x - 1) / filter_threadsPerBlock.x,
                             (fft_h + filter_threadsPerBlock.y - 1) / filter_threadsPerBlock.y);
    utils::float2complex<<<filter_blocksPerGrid, filter_threadsPerBlock>>>(
        fft_w, fft_h, padded_filter_ptr, d_padded_filter
    );
    dim3 output_threadsPerBlock(16, 16);
    dim3 output_blocksPerGrid((fft_w + output_threadsPerBlock.x - 1) / output_threadsPerBlock.x,
                             (fft_h + output_threadsPerBlock.y - 1) / output_threadsPerBlock.y);
    utils::float2complex<<<output_blocksPerGrid, output_threadsPerBlock>>>(
        fft_w, fft_h, padded_output_ptr, d_padded_output
    );
    cudaDeviceSynchronize();

    /**
        @note: Custom 2D Convolution
        DO NOT MODIFY DIMENSIONS HERE
    */
    //allocate device memory
    float *d_input, *d_filter, *d_output;
    cudaMalloc((void**)&d_input, size_of_input * sizeof(float));
    cudaMalloc((void**)&d_filter, size_of_filter * sizeof(float));
    cudaMalloc((void**)&d_output, out_width * out_height * sizeof(float));

    //copy data to device
    cudaMemcpy(d_input, h_input.data(), size_of_input * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_filter.data(), size_of_filter * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, h_output.data(), out_width * out_height * sizeof(float), cudaMemcpyHostToDevice);
    //launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((out_width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (out_height + threadsPerBlock.y - 1) / threadsPerBlock.y);
    // RUN KERNEL 
    cudaEventRecord(start);
    cuda_operations::_2DConv<<<blocksPerGrid, threadsPerBlock>>>(
        in_width, in_height, filter_width, filter_height, stride, padding,
        d_input, d_filter, d_output
    );
    cudaEventRecord(stop);
    
    // synchronize
    cudaDeviceSynchronize();
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Time: " << milliseconds << " ms" << std::endl;

    //copy result back to host
    cudaMemcpy(h_output.data(), d_output, out_width * out_height * sizeof(float), cudaMemcpyDeviceToHost);
    //free device memory
    cudaFree(d_input);
    cudaFree(d_filter);
    cudaFree(d_output);
    //print output
    std::cout << "Convolution Output: " << std::endl;
    for (int i = 0; i < out_height; ++i) {
        for (int j = 0; j < out_width; ++j) {
            std::cout << h_output[i * out_width + j] << " ";
        }
        std::cout << std::endl;
    }
    /**
        @note: End of Custom 2D Convolution
    */

    /**
        @note: Torch Conv2D Equivalence Test
    */
    std::cout << "Testing Torch Conv2D equivalence..." << std::endl;    
    // pass input,filter and to device
    input_tensor = input_tensor.to(device);
    filter_tensor = filter_tensor.to(device);
    cudaEventRecord(start);
    // RUN KERNEL 
    auto torchConv2d_result = torch::conv2d(
        input_tensor, filter_tensor, /*bias=*/{}, /*stride=*/{stride, stride},
        /*padding=*/{padding, padding}, /*dilation=*/{1, 1}, /*groups=*/1
    );
    // Stop timer
    cudaEventRecord(stop);
    
    std::cout << "torchConv2d_result : " << std::endl;
    std::cout << torchConv2d_result << std::endl;

    cudaDeviceSynchronize();
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Time: " << milliseconds << " ms" << std::endl;

    /**
        @note: End of Torch Conv2D Equivalence Test
    */

    /**
        @note: Spectral Conv2D Equivalence Test
    */
    // Call 2DFFTConv 
    std::cout << "Testing Spectral Conv2D equivalence..." << std::endl;
    cudaEventRecord(start);
    // RUN KERNEL
    cuda_operations::_2D_FFTConv(
        fft_h, fft_w, fft_h, fft_w,
        d_padded_input, d_padded_filter, d_padded_output
    );
    cudaEventRecord(stop);

    // synchronize
    cudaDeviceSynchronize();
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Time: " << milliseconds << " ms" << std::endl;


    //Calulate offset dims @note SINCE USING CROSS-CORRELATION - DESIRED OUTPUT IS LOCATED AT TOP-LEFT 
    int offset_w = 0;
    int offset_h = 0;
   
    // Convert back to float*
    utils::complex2float<<<output_blocksPerGrid, output_threadsPerBlock>>>(
        fft_w, fft_w, d_padded_output, padded_output_ptr
    );
    std::vector <float> spectral_output(out_width * out_height, 0.0f);
    cudaMemcpy2D(
        spectral_output.data(), //1. dst
        out_width * sizeof(float), // 2. dstPitch
        padded_output_ptr + (offset_h * fft_w + offset_w), // 3. src
        fft_w * sizeof(float), // 4. srcPitch
        out_width * sizeof(float), // 5. width
        out_height, // 6. height
        cudaMemcpyDeviceToHost // 7. kind
    );
    
    // Print the spectral output
    std::cout << "Spectral Conv2D Output : " << std::endl;
    for (int i = 0; i < out_height; ++i) {
        for (int j = 0; j < out_width; ++j) {
            std::cout << spectral_output[i * out_width + j] << " ";
        }
        std::cout << std::endl;
    }
    /**
        @note: End of Spectral Conv2D Equivalence Test
    */
    
    //free complex device memory
    cudaFree(d_padded_input);
    cudaFree(d_padded_filter);
    cudaFree(d_padded_output);
    cudaFree(d_input);
    cudaFree(d_filter);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    std::cout << "2D Convolution test executed." << std::endl;
}


/**
    @brief Main function to run convolution tests
**/
void convolution_test::convolve(){ 
    std::cout << "Starting various convolution tests..." << std::endl;
    //convolution_test::test1DConvolution();
    std::vector<int> input_dims = {32, 64};
    std::vector<int> filter_dims = {3, 5, 7, 11, 16};
    int stride = 1;
    int padding = 0;
    convolution_test::test2DConvolution(5, 5, 3, 3, stride, padding);
    std::cout << "All convolution tests completed." << std::endl;
}