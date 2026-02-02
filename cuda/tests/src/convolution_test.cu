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
template<bool image_test>
void convolution_test::test2DConvolution(
    int test_input_height,
    int test_input_width,
    int test_filter_height,
    int test_filter_width,
    int test_stride,
    int test_padding,
    cv::Mat test_image, 
    cv::Mat test_filter
    ){
    std ::cout << "Running 2D Convolution Test..." << std::endl;

    // Static clock
    cudaEvent_t start, stop;
    float custom_conv2d_milliseconds = 0;
    float torch_conv2d_milliseconds = 0;
    float spectral_conv2d_milliseconds = 0;
    float shared_memory_spectral_conv2d_milliseconds = 0;
    float cuFFT_conv2d_milliseconds = 0;
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

    // generate random input and filter
    
    std::vector<float> h_input(in_width * in_height);
    std::vector<float> h_filter(filter_width * filter_height);
    if(!image_test){
        std::cout << "Generating random input and filter..." << std::endl;
        for (int i = 0; i < in_width * in_height; ++i) {
        h_input[i] = static_cast<float>(rand() % 10); // values between 0 and 9
        }
        for (int i = 0; i < filter_width * filter_height; ++i) {
        h_filter[i] = static_cast<float>(rand() % 3 - 1); // values between -1 and 1
        }
    }
    else{
        std::cout << "Using provided image and filter for input..." << std::endl;
        // Convert cv::Mat to std::vector<float> for input
        h_input.resize(in_width * in_height);
        for (int i = 0; i < in_height; ++i) {
            for (int j = 0; j < in_width; ++j) {
                h_input[i * in_width + j] = static_cast<float>(test_image.at<uchar>(i, j));
            }
        }
        // Convert cv::Mat to std::vector<float> for filter
        h_filter.resize(filter_width * filter_height);
        for (int i = 0; i < filter_height; ++i) {
            for (int j = 0; j < filter_width; ++j) {
                h_filter[i * filter_width + j] = static_cast<float>(test_filter.at<float>(i, j));
            }
        }
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
    
    //Check padding dims 
    std::cout << "{Input} width to pad to: " << padded_in_width << " height to pad to: " << padded_in_height << std::endl;
    std::cout << "{Filter} width to pad to: " << padded_filter_width << " height to pad to: " << padded_filter_height << std::endl;
    std::cout << "{FFT Conv} Target dimensions: " << fft_w << " x " << fft_h << std::endl;
    
    // pad to target dimensions 
    torch::Tensor padded_input = torch::constant_pad_nd(input_tensor, {0, padded_in_width, 0, padded_in_height}, 0).contiguous();
    torch::Tensor padded_filter = torch::constant_pad_nd(filter_tensor, {0, padded_filter_width, 0, padded_filter_height}, 0).contiguous();
    torch::Tensor padded_output = torch::zeros({fft_h, fft_w}, gpu_options).contiguous();

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
    int max_threads = 32; // 4080 CUDA max threads per block is 1024
    dim3 fft_threadsPerBlock(max_threads, max_threads);
    dim3 fft_blocksPerGrid((fft_w + fft_threadsPerBlock.x - 1) / fft_threadsPerBlock.x,
                           (fft_h + fft_threadsPerBlock.y - 1) / fft_threadsPerBlock.y);
    cuda_operations::float2complex<<<fft_blocksPerGrid, fft_threadsPerBlock>>>(
        fft_w, fft_h, padded_input_ptr, d_padded_input
    );
    dim3 filter_threadsPerBlock(max_threads, max_threads);
    dim3 filter_blocksPerGrid((fft_w + filter_threadsPerBlock.x - 1) / filter_threadsPerBlock.x,
                             (fft_h + filter_threadsPerBlock.y - 1) / filter_threadsPerBlock.y);
    cuda_operations::float2complex<<<filter_blocksPerGrid, filter_threadsPerBlock>>>(
        fft_w, fft_h, padded_filter_ptr, d_padded_filter
    );
    dim3 output_threadsPerBlock(max_threads, max_threads);
    dim3 output_blocksPerGrid((fft_w + output_threadsPerBlock.x - 1) / output_threadsPerBlock.x,
                             (fft_h + output_threadsPerBlock.y - 1) / output_threadsPerBlock.y);
    cuda_operations::float2complex<<<output_blocksPerGrid, output_threadsPerBlock>>>(
        fft_w, fft_h, padded_output_ptr, d_padded_output
    );
    cudaDeviceSynchronize();

    // IMPORTANT SAVE STATE OF d_padded_input, d_padded_filter, d_padded_output FOR EACH TEST
    cuComplex* d_saved_padded_input, * d_saved_padded_filter;
    cudaMalloc((void**)&d_saved_padded_input, fft_h * fft_w * sizeof(cuComplex));
    cudaMalloc((void**)&d_saved_padded_filter, fft_h * fft_w * sizeof(cuComplex));

    cudaMemcpy(d_saved_padded_input, d_padded_input, fft_h * fft_w * sizeof(cuComplex), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_saved_padded_filter, d_padded_filter, fft_h * fft_w * sizeof(cuComplex), cudaMemcpyDeviceToDevice);

    // DEBUG: Check CuComplex pointers 
    // utils::checkcuComplexArray(d_padded_input, fft_w, fft_h, "Padded Input");
    // utils::checkcuComplexArray(d_padded_filter, fft_w, fft_h, "Padded Filter");
    // utils::checkcuComplexArray(d_padded_output, fft_w, fft_h, "Padded Output");
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
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    cuda_operations::_2DConv<<<blocksPerGrid, threadsPerBlock>>>(
        in_width, in_height, filter_width, filter_height, stride, padding,
        d_input, d_filter, d_output
    );
    // synchronize
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "Custom Conv2d Failed: " << cudaGetErrorString(err) << " at dim " << in_width * in_height << std::endl;
        return; // Exit before Torch tries to run and throws the AcceleratorError
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&custom_conv2d_milliseconds, start, stop);
    std::cout << "Custom 2D Convolution Time Test:" << std::endl;
    std::cout << "Time: " << custom_conv2d_milliseconds << " ms" << std::endl;

    //copy result back to host
    cudaMemcpy(h_output.data(), d_output, out_width * out_height * sizeof(float), cudaMemcpyDeviceToHost);
    //free device memory

    //print output
    std:: cout << "matrix dimensions: " << out_height << " x " << out_width << std::endl;
    std::cout << "Convolution Output: " << std::endl;
    //utils::printConvResult(h_output, out_width, out_height);
    /**
        @note: End of Custom 2D Convolution
    */

    /**
        @note: Torch Conv2D Equivalence Test
    */
    std::cout << "Testing Torch Conv2D equivalence..." << std::endl;    
    // Copy torch tensors to scratch 
    auto scratch_input = input_tensor.clone();
    auto scratch_filter = filter_tensor.clone();
    // Warm up Torch Conv2d
    auto warmup_torchConv2d = torch::conv2d(
        scratch_input, scratch_filter, /*bias=*/{}, /*stride=*/{stride, stride},
        /*padding=*/{padding, padding}, /*dilation=*/{1, 1}, /*groups=*/1
    );
    cudaDeviceSynchronize();

    // refresh input and filter tensors
    input_tensor = input_tensor.clone();
    filter_tensor = filter_tensor.clone();

    input_tensor = input_tensor.to(device);
    filter_tensor = filter_tensor.to(device);
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    // RUN KERNEL 
    auto torchConv2d_result = torch::conv2d(
        input_tensor, filter_tensor, /*bias=*/{}, /*stride=*/{stride, stride},
        /*padding=*/{padding, padding}, /*dilation=*/{1, 1}, /*groups=*/1
    );
    // Stop timer
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "Torch Kernel Failed: " << cudaGetErrorString(err) << " at dim " << in_width * in_height << std::endl;
        return; // Exit before Spectral tries to run and throws the AcceleratorError
    }     
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&torch_conv2d_milliseconds, start, stop);
    std::cout << "Time: " << torch_conv2d_milliseconds << " ms" << std::endl;

    /**
        @note: End of Torch Conv2D Equivalence Test
    */

    /**
        @note: Spectral Conv2D Equivalence Test
    */
    // Call 2DFFTConv 
    std::cout << "Testing Spectral Conv2D equivalence..." << std::endl;
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    // RUN KERNEL
    cuda_operations::_2D_FFTConv(
        fft_h, fft_w, fft_h, fft_w,
        d_padded_input, d_padded_filter, d_padded_output
    );
     // synchronize
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "Spectral Kernel Failed: " << cudaGetErrorString(err) << " at dim " << in_width * in_height << std::endl;
        return; // Exit before Torch tries to run and throws the AcceleratorError
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&spectral_conv2d_milliseconds, start, stop);
    std::cout << "Time: " << spectral_conv2d_milliseconds << " ms" << std::endl;


    //Calulate offset dims @note SINCE USING CROSS-CORRELATION - DESIRED OUTPUT IS LOCATED AT TOP-LEFT 
    int offset_w = 0; //filter_width - 1;
    int offset_h = 0; //filter_height - 1;
   
    // Convert back to float*
    cuda_operations::complex2float<<<output_blocksPerGrid, output_threadsPerBlock>>>(
        fft_w, fft_w, d_padded_output, padded_output_ptr
    );
    
    // Output for spectral conv2d
    std::vector <float> spectral_output(out_width * out_height, 0.0f);
    //Create a copy for cuFFT conv2d
    std::vector <float> spectral_output_cufft(out_width * out_height, 0.0f);
    // Output for Optimised spectral conv2d
    std::vector <float> optimised_spectral_output(out_width * out_height, 0.0f);


    // Copy relevant output region back to host
    cudaMemcpy2D(
        spectral_output.data(), //1. dst
        out_width * sizeof(float), // 2. dstPitch
        padded_output_ptr + (offset_h * fft_w + offset_w), // 3. src
        fft_w * sizeof(float), // 4. srcPitch
        out_width * sizeof(float), // 5. width
        out_height, // 6. height
        cudaMemcpyDeviceToHost // 7. kind
    );
    
    //Print the spectral output
    std::cout << "matrix dimensions: " << out_height << " x " << out_width << std::endl;
    std::cout << "Spectral Conv2D Output : " << std::endl;
    //utils::printConvResult(spectral_output, out_width, out_height);
    /**
        @note: End of Spectral Conv2D Equivalence Test
    */

    /**
        @note: Optimised FFTConv2D Equivalence Test
    */


    std::cout << "Testing Optimised FFT Conv2D equivalence..." << std::endl;
    // Warm up Optimised FFTConv
    // cuda_operations::Optimised2DFFTConv(
    //     fft_w, fft_h,
    //     d_padded_input, d_padded_filter, d_padded_output
    // );
    // cudaDeviceSynchronize();
    // // refresh input and filter tensors
    cudaMemcpy(d_padded_input, d_saved_padded_input, fft_h * fft_w * sizeof(cuComplex), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_padded_filter, d_saved_padded_filter, fft_h * fft_w * sizeof(cuComplex), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_padded_output, d_saved_padded_input, fft_h * fft_w * sizeof(cuComplex), cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();
    cudaEventRecord(start);
    // RUN KERNEL
    cuda_operations::Optimised2DFFTConv(
        fft_w, fft_h,
        d_padded_input, d_padded_filter, d_padded_output
    );
     // synchronize
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "Optimised FFT Kernel Failed: " << cudaGetErrorString(err)
                    << " at dim " << in_width * in_height << std::endl; 
        return; // Exit before Torch tries to run and throws the AcceleratorError
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&shared_memory_spectral_conv2d_milliseconds, start, stop);
    std::cout << "Time: " << shared_memory_spectral_conv2d_milliseconds << " ms" << std::endl;
    //Calulate offset dims @note SINCE USING CROSS-CORRELATION - DESIRED OUTPUT IS LOCATED AT TOP-LEFT
    // Convert back to float*
    cuda_operations::complex2float<<<output_blocksPerGrid, output_threadsPerBlock>>>(
        fft_w, fft_w, d_padded_output, padded_output_ptr
    );
    cudaMemcpy2D(
        optimised_spectral_output.data(), //1. dst
        out_width * sizeof(float), // 2. dstPitch
        padded_output_ptr + (offset_h * fft_w + offset_w), // 3. src
        fft_w * sizeof(float), // 4. srcPitch
        out_width * sizeof(float), // 5. width
        out_height, // 6. height
        cudaMemcpyDeviceToHost // 7. kind
    );
    std::cout << "matrix dimensions: " << out_height << " x " << out_width << std::endl;

    // Print the spectral output
    std::cout << "Optimised FFT Conv2D Output : " << std::endl;
    //utils::printConvResult(optimised_spectral_output, out_width, out_height);
    
    /**
        @note cuFFTConv Test

    */
    std::cout << "Testing cuFFT Conv2D equivalence..." << std::endl;    

    //Setup cuFFT based Conv2D
    cufftHandle plan;
    if(cufftPlan2d(&plan, fft_h, fft_w, CUFFT_C2C) != CUFFT_SUCCESS){
        std::cerr << "CUFFT Plan Creation Failed at dim: " << fft_h << " x " << fft_w << std::endl;
        return;
    }
    // Setup dummy memory ptrs
    cuComplex *d_input_scratch, *d_filter_scratch;
    cudaMalloc((void**)&d_input_scratch, fft_w * fft_h * sizeof(cuComplex));
    cudaMalloc((void**)&d_filter_scratch, fft_w * fft_h * sizeof(cuComplex));

    cudaMemcpy(d_input_scratch, d_saved_padded_input, fft_w * fft_h * sizeof(cuComplex), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_filter_scratch, d_saved_padded_filter, fft_w * fft_h* sizeof(cuComplex), cudaMemcpyDeviceToDevice);

    //Warm up cuFFT
    cuda_operations::_2DcuFFTConv(
        plan,
        fft_w, fft_h, fft_w, fft_h,
        d_input_scratch, d_filter_scratch, d_padded_output
    );
    cudaDeviceSynchronize();

    // refresh input and filter tensors
    cudaMemcpy(d_input_scratch, d_saved_padded_input, fft_w * fft_h * sizeof(cuComplex), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_filter_scratch, d_saved_padded_filter, fft_w * fft_h* sizeof(cuComplex), cudaMemcpyDeviceToDevice);
    cudaMemcpy(d_padded_output, d_saved_padded_input, fft_w * fft_h * sizeof(cuComplex), cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();
 
    // Start Timer
    cudaEventRecord(start);
    // RUN KERNEL
    cuda_operations::_2DcuFFTConv(
        plan,
        fft_w, fft_h, fft_w, fft_h,
        d_input_scratch, d_filter_scratch, d_padded_output
    );
     // synchronize
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        std::cerr << "cuFFT Kernel Failed: " << cudaGetErrorString(err)
                    << " at dim " << in_width * in_height << std::endl; 
        return; // Exit before Torch tries to run and throws the AcceleratorError
    }     
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&cuFFT_conv2d_milliseconds, start, stop);
    std::cout << "Time: " << cuFFT_conv2d_milliseconds << " ms" << std::endl;

    //Calulate offset dims @note SINCE USING CROSS-CORRELATION - DESIRED OUTPUT IS LOCATED AT TOP-LEFT 
  
    // Convert back to float*
    cuda_operations::complex2float<<<output_blocksPerGrid, output_threadsPerBlock>>>(
        fft_w, fft_w, d_padded_output, padded_output_ptr
    );
    
    cudaMemcpy2D(
        spectral_output_cufft.data(), //1. dst
        out_width * sizeof(float), // 2. dstPitch
        padded_output_ptr + (offset_h * fft_w + offset_w), // 3. src
        fft_w * sizeof(float), // 4. srcPitch
        out_width * sizeof(float), // 5. width
        out_height, // 6. height
        cudaMemcpyDeviceToHost // 7. kind
    );
    // Print the spectral output
    // std::cout << "cuFFT Conv2D Output : " << std::endl;
    // utils::printConvResult(spectral_output, out_width, out_height);


    //free complex device memory
    // cudaFree(d_padded_input);
    // cudaFree(d_padded_filter);
    // cudaFree(d_padded_output);

    cudaFree(d_input);
    cudaFree(d_filter);
    cudaFree(d_output);

    // free scratch pads
    cudaFree(d_input_scratch);
    cudaFree(d_filter_scratch);

    // destroy events and plan
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cufftDestroy(plan);
    std::cout << "2D Convolution test executed." << std::endl;

    
    cudaError_t err_final = cudaGetLastError();
    if (err_final != cudaSuccess) {
        std::cerr << "Post-test CUDA error: " << cudaGetErrorString(err_final) << std::endl;
    }

    //copy h_output and spectral output to float vectors for error measurement
    std::vector<float> h_output_vec(h_output.begin(), h_output.end());
    std::vector<float> spectral_output_vec(spectral_output.begin(), spectral_output.end());
    std::vector<float> optimised_spectral_output_vec(optimised_spectral_output.begin(), optimised_spectral_output.end());
    
    // measure correctness between methods
    float mse = utils::MeasureError(
        h_output_vec,
        spectral_output_vec
    );
    //measure correctness between custom and optimised spectral
    float mse_optimised = utils::MeasureError(
        h_output_vec,
        optimised_spectral_output_vec
    );
    //Measure correctness between custom and cuFFT
    float mse_cufft = utils::MeasureError(
        h_output_vec,
        spectral_output_cufft
    );

    // Print MSE results
    // std::cout << "Mean Squared Error between Custom 2D Conv and Spectral 2D FFT Conv: " << mse << std::endl;
    // std::cout << "Mean Squared Error between Custom 2D Conv and Optimised Spectral 2D FFT Conv: " << mse_optimised << std::endl;
    // std::cout << "Mean Squared Error between Custom 2D Conv and cuFFT 2D Conv: " << mse_cufft << std::endl;


    // write results to result_file 
    /**
        Conv_Method | Input_Dimensions | Filter_Dimensions | Stride | Padding | Time_ms 
    */
    std::filesystem::path root = std::filesystem::current_path().parent_path().parent_path().parent_path();
    std::string path_to_results = "/data/measurements/";
    std::string test_name = image_test ? "image_" + std::to_string(in_height) + "x" + std::to_string(in_width) 
    : "random_test" + std::to_string(in_height) + "x" + std::to_string(in_width) + "_" + std::to_string(filter_height) + "x" + std::to_string(filter_width);
    std::string result_file = test_name + "_2DConv_results_Sobel.csv";
    
    std::vector<std::string> Runtime_csv_header = {
        "Conv_Method",
        "Input_Dimensions",
        "Filter_Dimensions",
        "Stride",
        "Padding",
        "Time_ms"
    };
    std::vector<std::string> Runtime_csv_content = {
        "Custom_2DConv",
        std::to_string(in_height) + "x" + std::to_string(in_width),
        std::to_string(filter_height) + "x" +  std::to_string(filter_width),
        std::to_string(stride),
        std::to_string(padding),
        std::to_string(custom_conv2d_milliseconds),
        "Torch_Conv2D",
        std::to_string(in_height) + "x" + std::to_string(in_width),
        std::to_string(filter_height) + "x" +  std::to_string(filter_width),
        std::to_string(stride),
        std::to_string(padding),
        std::to_string(torch_conv2d_milliseconds),
        "Spectral_2DFFTConv",
        std::to_string(in_height) + "x" + std::to_string(in_width),
        std::to_string(filter_height) + "x" +  std::to_string(filter_width),
        std::to_string(stride),
        std::to_string(padding),
        std::to_string(spectral_conv2d_milliseconds),
        "OptimisedSharedMemory_2DFFTConv",
        std::to_string(in_height) + "x" + std::to_string(in_width),
        std::to_string(filter_height) + "x" +  std::to_string(filter_width),
        std::to_string(stride),
        std::to_string(padding),
        std::to_string(shared_memory_spectral_conv2d_milliseconds),
        "CuFFT_2DConv",
        std::to_string(in_height) + "x" + std::to_string(in_width),
        std::to_string(filter_height) + "x" +  std::to_string(filter_width),
        std::to_string(stride),
        std::to_string(padding),
        std::to_string(cuFFT_conv2d_milliseconds)
    };
    std::string Runtime_entire_path = root.string() + path_to_results + result_file;
    utils::writeCSV(Runtime_entire_path, Runtime_csv_content, Runtime_csv_header);

    // MSE Results
    std::vector<std::string> mse_csv_header = {
        "Method", 
        "Input_Dimensions",
        "Filter_Dimensions",
        "MSE_ERROR_VS_Custom_2DConv"

    };
    std::vector<std::string> mse_csv_content = {
        "Spectral_2DFFTConv",
        std::to_string(in_height) + "x" + std::to_string(in_width),
        std::to_string(filter_height) + "x" +  std::to_string(filter_width),
        std::to_string(mse),
        "OptimisedSharedMemory_2DFFTConv",
        std::to_string(in_height) + "x" + std::to_string(in_width),
        std::to_string(filter_height) + "x" +  std::to_string(filter_width),
        std::to_string(mse_optimised),
        "CuFFT_2DConv",
        std::to_string(in_height) + "x" + std::to_string(in_width),
        std::to_string(filter_height) + "x" +  std::to_string(filter_width),
        std::to_string(mse_cufft)
    };
    std::string error_result_file = test_name + "_2DConv_MSE_results_sobel.csv";
    std::string error_entire_path = root.string() + path_to_results + error_result_file;
    utils::writeCSV(error_entire_path, mse_csv_content, mse_csv_header);
    if(image_test){
        // save output images of each method for visual comparison
        utils::saveOutputImage("custom_2dconv_output_sobel.png", h_output_vec, out_width, out_height);
        utils::saveOutputImage("spectral_2dfftconv_output_sobel.png", spectral_output_vec, out_width, out_height);
        utils::saveOutputImage("optimised_spectral_2dfftconv_output_sobel.png", optimised_spectral_output_vec, out_width, out_height);
        utils::saveOutputImage("cufft_2dconv_output_sobel.png", spectral_output_cufft, out_width, out_height);
    }
}


/**
    @brief Main function to run convolution tests
**/
void convolution_test::convolve(char* argv[]){ 
    std::cout << "Starting various convolution tests..." << std::endl;
    //convolution_test::test1DConvolution();
    // explodes for conv over 16 need to fix 
    // Alex net convs are 32x32,11x11,5x5,3x3

    // provide vectors 128, 256, 512, 1024
    // Input dims to test {32(CIFAR-10), 64, 128, 224 (ImageNet), 256, 512, 1024}
    // Filter dims to test {5, 16, 32, 64, 128, 256, 512, 1024}
    
   
    TestMode mode = parseMode(argv[2]);
    
    std::vector<int> input_dims = {2048};
    std::vector<int> filter_dims = {3,5,11,16,32,64,128,256,512,1024,2048};
    int stride = 1;
    int padding = 0;

    if (mode == TestMode::Image) {
        std::cout << "Running 2D Convolution Test on Image..." << std::endl;
        // Load image using OpenCV
        std::string data_path = std::filesystem::current_path().parent_path().parent_path().parent_path().string() + "/data/userdata/";

        std::string image_path =  data_path + "cat.png";
        cv::Mat image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
        if (image.empty()) {
            std::cerr << "Error: Could not load image at " << image_path << std::endl;
            return;
        }
        std::cout << "Image loaded with dimensions: " << image.rows << " x " << image.cols << std::endl;
        // Define a sample filter (e.g., edge detection)
        cv::Mat filter = (cv::Mat_<float>(3,3) <<
            -1, 0, 1,
            -2,  0, 2,
            -1, 0, 1
        );
        // Run 2D convolution test with the image and filter
        convolution_test::test2DConvolution<true>(
            image.rows,
            image.cols,
            filter.rows,
            filter.cols,
            stride,
            padding,
            image,
            filter
        );
    }
    else{
        std::cout << "Running 2D Convolution Tests on Random Inputs..." << std::endl;
        for(const auto& in_dim : input_dims){
            for(const auto& filter_dim : filter_dims){
                convolution_test::test2DConvolution<false>(
                    in_dim,
                    in_dim,
                    filter_dim,
                    filter_dim,
                    stride,
                    padding,
                    cv::Mat(),
                    cv::Mat()
                );
            }
        }
    }
   
    //convolution_test::test2DConvolution(128,128, 64, 64, stride, padding); // Input {3,6,11,} filter{3, 5, 7} 
    std::cout << "All convolution tests completed." << std::endl;
}