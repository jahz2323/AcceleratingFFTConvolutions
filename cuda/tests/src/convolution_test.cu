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
**/
void convolution_test::test2DConvolution(){
    std ::cout << "Running 2D Convolution Test..." << std::endl;
    // Implementation for 2D convolution test would go here
    int in_width = 4;
    int in_height = 4;
    int filter_width = 3;
    int filter_height = 3;
    int stride = 1;
    int padding = 1;
    int out_width = ((in_width - filter_width + 2 * padding) / stride) + 1;
    int out_height = ((in_height - filter_height + 2 * padding) / stride) + 1;

    //allocate and initialize host memory
    std::vector<float> h_input = {
        1, 2, 3, 4,
        5, 6, 7, 8,
        9,10,11,12,
       13,14,15,16
    };
    std::vector<float> h_filter = {
        1,0,1,
        0,1,0,
        1,0,1
    };
    std::vector<float> h_output(out_width * out_height, 0.0f);

    //allocate device memory
    float *d_input, *d_filter, *d_output;
    cudaMalloc((void**)&d_input, in_width * in_height * sizeof(float));
    cudaMalloc((void**)&d_filter, filter_width * filter_height * sizeof(float));
    cudaMalloc((void**)&d_output, out_width * out_height * sizeof(float));

    //copy data to device
    cudaMemcpy(d_input, h_input.data(), in_width * in_height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_filter, h_filter.data(), filter_width * filter_height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_output, h_output.data(), out_width * out_height * sizeof(float), cudaMemcpyHostToDevice);

    //launch kernel
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((out_width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (out_height + threadsPerBlock.y - 1) / threadsPerBlock.y);
    cuda_operations::_2DConv<<<blocksPerGrid, threadsPerBlock>>>(
        in_width, in_height, filter_width, filter_height, stride, padding,
        d_input, d_filter, d_output
    );
    cudaDeviceSynchronize();

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

    std::cout << "Testing Torch Conv2D equivalence..." << std::endl;
    auto input_tensor = torch::from_blob(h_input.data(), {1, 1, in_height, in_width}).clone();
    auto filter_tensor = torch::from_blob(h_filter.data(), {1, 1, filter_height, filter_width}).clone();
    auto output_tensor = torch::conv2d(input_tensor, filter_tensor, {}, stride, padding);
    std::cout << "Torch Conv2D Output : " << std::endl;
    std::cout << output_tensor << std::endl;


    std::cout << "2D Convolution test executed." << std::endl;
}


/**
    @brief Main function to run convolution tests
**/
void convolution_test::convolve(){ 
    std::cout << "Starting various convolution tests..." << std::endl;
    //convolution_test::test1DConvolution();
    //convolution_test::test2DConvolution();
    std::cout << "All convolution tests completed." << std::endl;
}