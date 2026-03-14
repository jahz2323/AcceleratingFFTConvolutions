#include "convolution_test.cuh"
#include "dataprep.hpp"
#include <nvtx3/nvToolsExt.h> // For NVTX profiling ranges

void convolution_test::memory_usage(
    ConvContext& ctx,
    const std::string& full_path, 
    size_t Direct_Conv2D_memory_bytes,
    size_t FFT_Conv2D_memory_bytes,
    size_t FFT_OVA_Conv2D_memory_bytes // incase not called default = 0
){
    /*
        CTX:
        int in_h, in_w, f_h, f_w, stride, pad; 
        int out_h, out_w;
        int fft_h, fft_w;
    */
    
    std::vector<std::string> memory_usage_header = {
        "Method",
        "Input_Dimensions",
        "Filter_Dimensions",
        "Memory_usage_in_bytes"
    };

    std::vector<std::string> memory_usage_contnet = {
        "Direct_Conv2D",
        std::to_string(ctx.in_h) + "x" + std::to_string(ctx.in_w),
        std::to_string(ctx.f_h) + "x" +  std::to_string(ctx.f_w),
        std::to_string(Direct_Conv2D_memory_bytes),
        "FFT_Conv2D",
        std::to_string(ctx.in_h) + "x" + std::to_string(ctx.in_w),
        std::to_string(ctx.f_h) + "x" +  std::to_string(ctx.f_w),
        std::to_string(FFT_Conv2D_memory_bytes),
        "FFT_OVA_Conv2D",
        std::to_string(ctx.in_h) + "x" + std::to_string(ctx.in_w),
        std::to_string(ctx.f_h) + "x" +  std::to_string(ctx.f_w),
        std::to_string(FFT_OVA_Conv2D_memory_bytes),
    };

    std::filesystem::path base_path(full_path);
    std::string csv_path = base_path.string() + ".csv";
     // Get directory path build/xxx/
    std::filesystem::path dir = base_path.parent_path();
    std::filesystem::path memory_dir = dir / "results/memory";
    std::string memory_csv_name = (full_path + (".csv"));

    // Write MSE results to CSV file, if file already exists, append new results as a new row, if not create new file and write header and results, if csv is not present, create a new file. If csv is present, check if header is present, if not write header, then append results as a new row. If csv is present and header is present, just append results as a new row.
    utils::writeCSV(memory_csv_name, memory_usage_contnet, memory_usage_header);
}


void convolution_test::mse(
    ConvContext& ctx,
    bool image_test,
    const std::string& full_path,
    float mse_val,
    float mse_optimised,
    float mse_cufft
)
{
    int in_height = ctx.in_h;
    int in_width = ctx.in_w;
    int filter_height = ctx.f_h;
    int filter_width = ctx.f_w;
 

    // MSE Results
    std::vector<std::string> mse_csv_header = {
        "Method", 
        "Input_Dimensions",
        "Filter_Dimensions",
        "MSE_ERROR_VS_Custom_2DConv"

    };
    std::vector<std::string> mse_csv_content = {
        ctx.use_ova ? "FFT_OVA_Conv2D" : "Spectral_Conv2D",
        std::to_string(in_height) + "x" + std::to_string(in_width),
        std::to_string(filter_height) + "x" +  std::to_string(filter_width),
        std::to_string(mse_val),
    }; 

    std::filesystem::path base_path(full_path);

    std::string csv_path = base_path.string() + ".csv";
     // Get directory path build/xxx/
    std::filesystem::path dir = base_path.parent_path();
    std::filesystem::path images_dir = dir / "results/images"; 
    std::filesystem::path mse_dir = dir / "results/mse";

    std::string stem = base_path.filename().string();
    std::string mse_csv_name = (dir / ("MSE_" + stem + ".csv")).string();

    // Write MSE results to CSV file, if file already exists, append new results as a new row, if not create new file and write header and results, if csv is not present, create a new file. If csv is present, check if header is present, if not write header, then append results as a new row. If csv is present and header is present, just append results as a new row.
    utils::writeCSV(mse_csv_name, mse_csv_content, mse_csv_header);
}

void convolution_test::runtime(
    ConvContext& ctx, 
    bool image_test,
    const std::string& full_path,
    float custom_conv2d_milliseconds, 
    float spectral_conv2d_milliseconds, 
    float torch_conv2d_milliseconds,
    float shared_memory_spectral_conv2d_milliseconds,
    float cuFFT_conv2d_milliseconds
)
{

    int in_height = ctx.in_h;
    int in_width = ctx.in_w;
    int filter_height = ctx.f_h;
    int filter_width = ctx.f_w;
    int stride = ctx.stride;
    int padding = ctx.pad;

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
        ctx.use_ova ? "FFT_OVA_Conv2D" : "Spectral_Conv2D",
        std::to_string(in_height) + "x" + std::to_string(in_width),
        std::to_string(filter_height) + "x" +  std::to_string(filter_width),
        std::to_string(stride),
        std::to_string(padding),
        std::to_string(spectral_conv2d_milliseconds),
    };
   
    /**
        Results Structure 
        RUNTIME: 
            Runtime_ImageName_FilterSize_ **.csv
        MSE:
            MSE_ImageName_FilterSize_ ** .csv
        CONVOLUTION:
            CONV_METHOD_ImageName_FilterSize_ **.png
    */
    std::filesystem::path base_path(full_path);
    std::filesystem::path dir = base_path.parent_path();

    std::string stem = base_path.filename().string();

    std::string runtime_csv_name = (dir / ("Runtime_" + stem + ".csv")).string();

    std::cout << "Saving runtime results to \n: " << runtime_csv_name << std::endl;

    utils::writeCSV(runtime_csv_name, Runtime_csv_content, Runtime_csv_header);
};


//free context 
void convolution_test::freeContext(ConvContext& ctx){
    ctx.t_input = torch::Tensor(); 
    ctx.t_filter = torch::Tensor();
    ctx.t_output = torch::Tensor();
    if (ctx.use_ova) {
        // 1. Free the individual workspaces and scratches allocated per stream
        for (int i = 0; i < ctx.num_streams; ++i) {
            if (ctx.d_workspaces[i]) cudaFree(ctx.d_workspaces[i]);
            if (ctx.d_scratches[i])  cudaFree(ctx.d_scratches[i]);
            
            // 2. Destroy the stream handle
            cudaStreamDestroy(ctx.streams[i]);
            ctx.streams[i] = nullptr;
            ctx.workspace_block = nullptr;
            ctx.d_padded_filter_OVA = nullptr;
        }
        //3. Remove OVA Specific buffers
        cudaFree(ctx.workspace_block);
        cudaFree(ctx.d_padded_filter_OVA);
    }
    
    //2. Clear pytorch tensors to free associated GPU memory
    ctx.t_input = torch::Tensor();
    ctx.t_filter = torch::Tensor();
    ctx.t_output = torch::Tensor();

    // 3. Free the standard individual pointers
    cudaFree(ctx.d_input_float);
    cudaFree(ctx.d_filter_float);
    cudaFree(ctx.d_output_float);
    cudaFree(ctx.d_input_complex);
    cudaFree(ctx.d_filter_complex);
    cudaFree(ctx.d_output_complex);
    cudaFree(ctx.d_fft_output_float);
    cudaFree(ctx.d_saved_filter_complex);
    cudaFree(ctx.d_saved_input_complex);
    cudaFree(ctx.d_saved_filter_float);
    cudaFree(ctx.d_saved_input_float);
    

    //4. Reset all pointers to nullptr for safety
    ctx.d_input_float = nullptr;
    ctx.d_filter_float = nullptr;
    ctx.d_output_float = nullptr;
    ctx.d_input_complex = nullptr;
    ctx.d_filter_complex = nullptr;
    ctx.d_output_complex = nullptr;
    ctx.d_fft_output_float = nullptr;
    ctx.d_saved_filter_complex = nullptr;
    ctx.d_saved_input_complex = nullptr;
    ctx.d_saved_filter_float = nullptr;
    ctx.d_saved_input_float = nullptr;
    ctx.results.clear();
}

//CuFFT Runner 
void convolution_test::run_cuFFT(ConvContext& ctx)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    cufftHandle plan;
    cufftPlan2d(&plan, ctx.fft_h, ctx.fft_w, CUFFT_C2C);

    cudaEventRecord(start);
    // RUN KERNEL
    cuda_operations::_2DcuFFTConv(
        plan,
        ctx.fft_w, ctx.fft_h, ctx.fft_w, ctx.fft_h,
        ctx.d_input_complex, ctx.d_filter_complex, ctx.d_output_complex
    );
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ctx.results["cuFFTConv"].time_ms, start, stop);
    cufftDestroy(plan);
}

//StandardConv2D Runner
void convolution_test::run_direct(ConvContext& ctx)
{   
    //CLEAR OUTPUT STATE 
    cudaMemset(ctx.d_output_float, 0, ctx.out_w * ctx.out_h * sizeof(float)); // CLEAR PREVIOUS DATA

    std::cout << "-------- Running Custom Direct Convolution Kernel --------" << std::endl;
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    nvtxRangePushA("Direct_Convolution_Runner");
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((ctx.out_w + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (ctx.out_h + threadsPerBlock.y - 1) / threadsPerBlock.y);
    cudaEventRecord(start);

    // RUN KERNEL 
    cuda_operations::_2DConv<<<blocksPerGrid, threadsPerBlock>>>(
        ctx.in_w, ctx.in_h, ctx.f_w, ctx.f_h, ctx.stride, ctx.pad,
        ctx.d_input_float, ctx.d_filter_float, ctx.d_output_float
    );
    cudaError_t err = cudaDeviceSynchronize(); 
    if (err != cudaSuccess) {
        printf("KERNEL CRASHED: %s\n", cudaGetErrorString(err));
    }
    // == Measuring RUNTIME using NSIGHT ==
    cudaDeviceSynchronize();
    nvtxRangePop();

    std::vector<float> direct_conv_output(ctx.out_w * ctx.out_h, 0.0f);
    cudaMemcpy(direct_conv_output.data(), ctx.d_output_float, ctx.out_w * ctx.out_h * sizeof(float), cudaMemcpyDeviceToHost);
    //save to ctx.results for MSE calculation and image saving
    ctx.results["Custom_2DConv"].data = direct_conv_output;

    //utils::printConvResult(direct_conv_output, ctx.out_w, ctx.out_h);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ctx.results["Custom_2DConv"].time_ms, start, stop);
}

//Torch Conv2D Runner
void convolution_test::run_torch(ConvContext& ctx)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    std::cout << "-------- Running Torch Conv2D --------" << std::endl;
  
    // RUN KERNEL
    cudaEventRecord(start);
    ctx.t_output = torch::conv2d(
        ctx.t_input, ctx.t_filter, /*bias=*/{}, /*stride=*/{ctx.stride, ctx.stride},
        /*padding=*/{ctx.pad, ctx.pad}, /*dilation=*/{1, 1}, /*groups=*/1
    );
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ctx.results["Torch_Conv2D"].time_ms, start, stop);
}

//Spectral Conv2D Runner
void convolution_test::run_FFTConv(ConvContext& ctx)
{       
   
    // == restore from scratch for fair comparison ==
    // 1. Reset complex buffers to saved state before running spectral conv2d, to ensure same input data for both spectral methods and fair MSE comparison, since FFT-based convolution is non-deterministic due to floating point precision issues and different execution paths in the kernel
    cudaMemcpy(ctx.d_input_complex, ctx.d_saved_input_complex, ctx.fft_w * ctx.fft_h * sizeof(cuComplex), cudaMemcpyDeviceToDevice);
    cudaMemcpy(ctx.d_filter_complex, ctx.d_saved_filter_complex, ctx.fft_w * ctx.fft_h * sizeof(cuComplex), cudaMemcpyDeviceToDevice);
    //2. Clear output 
    cudaMemset(ctx.d_output_complex, 0, ctx.fft_w * ctx.fft_h * sizeof(cuComplex));
    std::cout << "-------- Running Custom FFT Convolution Kernel --------" << std::endl;
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);
    cudaEventRecord(start);
    // RUN KERNEL
    nvtxRangePushA("Spectral_Conv2D_Runner");
    if (ctx.use_ova){
        std::cout << "Running FFT-OVA Convolution with target block size of " << ctx.target_ova_size << std::endl;
        // Call OVA Conv 
        cuda_operations::FFT_OVA_Conv(
            ctx.in_w, ctx.in_h, ctx.f_w, ctx.f_h,
            ctx.stride, ctx.pad,
            ctx.num_streams,
            ctx.streams,
            ctx.d_input_complex, ctx.d_filter_complex, 
            ctx.d_output_complex, ctx.workspace_block,
            ctx.d_workspaces, ctx.d_scratches,
            ctx.segment_w, ctx.segment_h,
            ctx.block_w, ctx.block_h,
            ctx.num_blocks_w, ctx.num_blocks_h,
            ctx.total_blocks
        );
        //check size of output complex buffer to ensure it is correct for OVA conv output
        //int output_complex_size = ctx.out_w * ctx.out_h * sizeof(cuComplex);
        // std::cout << "DEBUG: Output complex buffer size (bytes): " << output_complex_size << std::endl;
        // std::vector<cuComplex> output_check(ctx.out_w * ctx.out_h);
        // cudaMemcpy(output_check.data(), ctx.d_output_complex, output_complex_size, cudaMemcpyDeviceToHost);
        // //print first 25 elements of output complex buffer for debugging
        // std::cout << "DEBUG: Output complex buffer sample FIRST 25 Elements: " << std::endl;
        // for(int i = 0; i < 5  * 5  ; i++){
        //     std::cout << output_check[i].x << "i" << output_check[i].y << "j " << std::endl;
        // }
        //convolution_test::run_OptimisedFFTConv(ctx); // for testing optimised spectral conv variant
         // == Measuring RUNTIME using NSIGHT ==
        cudaDeviceSynchronize();
        nvtxRangePop();
        
        //Store in ctx.results["Spectral_Conv2D"].data 
        nvtxRangePushA("Copy_Spectral_Output_and_Convert");
        //std::vector<cuComplex> fft_conv_output_complex(ctx.fft_w * ctx.fft_h);
        //cudaMemcpy(fft_conv_output_complex.data(), ctx.d_output_complex, ctx.fft_w * ctx.fft_h * sizeof(cuComplex), cudaMemcpyDeviceToHost);
        dim3 convert_threadsPerBlock(32, 32);
        dim3 convert_blocksPerGrid((ctx.out_w + convert_threadsPerBlock.x - 1) / convert_threadsPerBlock.x,
                                (ctx.out_h + convert_threadsPerBlock.y - 1) / convert_threadsPerBlock.y);
        //convert FFT output from complex to real and copy back to host
        cuda_operations::complex2float<<<convert_blocksPerGrid, convert_threadsPerBlock>>>(
            ctx.out_w, ctx.out_h, ctx.d_output_complex, ctx.d_fft_output_float
        );
        cudaDeviceSynchronize();
        nvtxRangePop();
        // Output for spectral conv2d
        std::vector<float> spectral_output(ctx.out_w * ctx.out_h, 0.0f);
        //Calulate offset dims @note SINCE USING CROSS-CORRELATION - DESIRED OUTPUT IS LOCATED AT TOP-LEFT 
        
        // Copy relevant output region back to host
        cudaMemcpy2D(
            spectral_output.data(), //1. dst
            ctx.out_w * sizeof(float), // 2. dstPitch
            ctx.d_fft_output_float, // 3. src which is cuComplex, but we want to copy real parts to float output, so we can treat it as float pointer with appropriate offset
            ctx.out_w * sizeof(float), // 4. srcPitch
            ctx.out_w * sizeof(float), // 5. width
            ctx.out_h, // 6. height
            cudaMemcpyDeviceToHost // 7. kind
        );
        // spectral_output -> ctx.results["Spectral_Conv2D"].data for MSE calculation and saving output image
        ctx.results["FFT_OVA_Conv2D"].data = spectral_output; 
        //std::cout << "Sample of spectral conv output data: " << spectral_output[0] << ", " << spectral_output[ctx.out_w * ctx.out_h / 2] << ", " << spectral_output[ctx.out_w * ctx.out_h - 1] << std::endl;
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ctx.results["FFT_OVA_Conv2D"].time_ms, start, stop);
    }
    else{
        // Standard FFT Conv
        cuda_operations::_2D_FFTConv(
            ctx.fft_h, ctx.fft_w, ctx.fft_h, ctx.fft_w,
            ctx.d_input_complex, ctx.d_filter_complex, ctx.d_output_complex
        );
         // == Measuring RUNTIME using NSIGHT ==
        cudaDeviceSynchronize();
        nvtxRangePop();
        
        //Store in ctx.results["Spectral_Conv2D"].data 
        nvtxRangePushA("Copy_Spectral_Output_and_Convert");
        std::vector<cuComplex> fft_conv_output_complex(ctx.fft_w * ctx.fft_h);
        cudaMemcpy(fft_conv_output_complex.data(), ctx.d_output_complex, ctx.fft_w * ctx.fft_h * sizeof(cuComplex), cudaMemcpyDeviceToHost);
        dim3 convert_threadsPerBlock(32, 32);
        dim3 convert_blocksPerGrid((ctx.fft_w + convert_threadsPerBlock.x - 1) / convert_threadsPerBlock.x,
                                (ctx.fft_h + convert_threadsPerBlock.y - 1) / convert_threadsPerBlock.y);
        //convert FFT output from complex to real and copy back to host
        cuda_operations::complex2float<<<convert_blocksPerGrid, convert_threadsPerBlock>>>(
            ctx.fft_w, ctx.fft_h, ctx.d_output_complex, ctx.d_fft_output_float
        );
        cudaDeviceSynchronize();
        nvtxRangePop();
        // Output for spectral conv2d
        std::vector<float> spectral_output(ctx.out_w * ctx.out_h, 0.0f);
        //Calulate offset dims @note SINCE USING CROSS-CORRELATION - DESIRED OUTPUT IS LOCATED AT TOP-LEFT 
        
        // Copy relevant output region back to host
        cudaMemcpy2D(
            spectral_output.data(), //1. dst
            ctx.out_w * sizeof(float), // 2. dstPitch
            ctx.d_fft_output_float, // 3. src which is cuComplex, but we want to copy real parts to float output, so we can treat it as float pointer with appropriate offset
            ctx.fft_w * sizeof(float), // 4. srcPitch
            ctx.out_w * sizeof(float), // 5. width
            ctx.out_h, // 6. height
            cudaMemcpyDeviceToHost // 7. kind
        );
        // spectral_output -> ctx.results["Spectral_Conv2D"].data for MSE calculation and saving output image
        ctx.results["Spectral_Conv2D"].data = spectral_output; 
        //std::cout << "Sample of spectral conv output data: " << spectral_output[0] << ", " << spectral_output[ctx.out_w * ctx.out_h / 2] << ", " << spectral_output[ctx.out_w * ctx.out_h - 1] << std::endl;
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&ctx.results["Spectral_Conv2D"].time_ms, start, stop);
    }

    
}
//Optimised Spectral Conv2D Runner
void convolution_test::run_OptimisedFFTConv(ConvContext& ctx)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    cudaEventRecord(start);
    // RUN KERNEL
    cuda_operations::Optimised2DFFTConv(ctx.in_w, ctx.in_h, ctx.d_input_complex, ctx.d_filter_complex, ctx.d_output_complex);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ctx.results["OptimisedSharedMemory_Spectral_Conv2D"].time_ms, start, stop);
}

void convolution_test::runStandardPooling(ConvContext& ctx)
{
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    std::cout << "-------- Running Custom Pooling Kernel --------" << std::endl;
    cudaEventRecord(start);
    // RUN KERNEL
    dim3 threadsPerBlock(16, 16);
    dim3 blocksPerGrid((ctx.out_w + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (ctx.out_h + threadsPerBlock.y - 1) / threadsPerBlock.y);
    cuda_operations::_2DPool<<<blocksPerGrid, threadsPerBlock>>>(
        ctx.in_w, ctx.in_h, 
        ctx.f_w, ctx.f_h, 
        ctx.stride, ctx.pad,
        ctx.d_input_float, ctx.d_output_float, 
        ctx.pool_mode
        
    );
    cudaError_t err = cudaDeviceSynchronize(); 
    if (err != cudaSuccess) {
        printf("POOLING KERNEL CRASHED: %s\n", cudaGetErrorString(err));
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ctx.results["Standard_Pooling"].time_ms, start, stop);
}

void convolution_test::resetFFTContext(ConvContext& ctx){
    // Utility function to reset complex buffers to saved state for fair runtime comparisons between spectral methods
    cudaMemcpy(ctx.d_input_complex, ctx.d_saved_input_complex, ctx.fft_w * ctx.fft_h * sizeof(cuComplex), cudaMemcpyDeviceToDevice);
    cudaMemcpy(ctx.d_filter_complex, ctx.d_saved_filter_complex, ctx.fft_w * ctx.fft_h * sizeof(cuComplex), cudaMemcpyDeviceToDevice);
    cudaMemset(ctx.d_output_complex, 0, ctx.fft_w * ctx.fft_h * sizeof(cuComplex));
}


template<bool isPooling>
void convolution_test::initaliseContext(
    ConvContext& ctx, 
    int in_h, int in_w, 
    int f_h, int f_w, 
    int stride, int pad,
    bool image_test, 
    cv::Mat test_image, cv::Mat test_filter, 
    int target_block_size,  bool use_ova
){
    
     //1. Set dimensions and calculate derived dimensions
    ctx.in_h = in_h; ctx.in_w = in_w; 
    ctx.f_h = f_h; ctx.f_w = f_w;
    ctx.stride = stride; ctx.pad = pad;
    ctx.out_h = ((in_h - f_h + 2 * pad) / stride) + 1;
    ctx.out_w = ((in_w - f_w + 2 * pad) / stride) + 1;

    // Load data: 
    //If Image test . Load data into host vectors - either random or from provided images
    if (!isPooling){
        // Standard FFT-Conv parameter
        if (!use_ova){
            //2. Calculate FFT dimensions
            int target_dimensions_height = in_h + f_h - 1;
            int target_dimensions_width = in_w + f_w - 1;
            ctx.fft_h = utils::nextPowerOfTwo(target_dimensions_height);
            ctx.fft_w = utils::nextPowerOfTwo(target_dimensions_width);

            ctx.h_input.resize(in_w * in_h);
            ctx.h_filter.resize(f_w * f_h);
            //3. Allocate and initialize host memory for input, filter and output
            if(!image_test){
                std::cout << "Generating random input and filter..." << std::endl;
                for (int i = 0; i < in_w * in_h; ++i) {
                    ctx.h_input[i] = static_cast<float>(rand() % 10); // values between 0 and 9
                }
                for (int i = 0; i < f_w * f_h; ++i) {
                    ctx.h_filter[i] = static_cast<float>(rand() % 3 - 1); // values between -1 and 1
                }
            }
            else{
                std::cout << "Using provided image and filter for input..." << std::endl;
                for (int i = 0; i < in_h; ++i) {
                    for (int j = 0; j < in_w; ++j) {
                        ctx.h_input[i * in_w + j] = static_cast<float>(test_image.at<uchar>(i, j));
                    }
                }
                // Convert cv::Mat to std::vector<float> for filter
                for (int i = 0; i < f_h; ++i) {
                    for (int j = 0; j < f_w; ++j) {
                        ctx.h_filter[i * f_w + j] = static_cast<float>(test_filter.at<float>(i, j));
                    }
                }
            }
        }
        // OVA-specific dims and parameters
        else {
            ctx.fft_h = ctx.target_ova_size; 
            ctx.fft_w = ctx.target_ova_size;

            std::cout << "Entry to FFT-OVA-Conv initalisation" << std::endl;
            // Data is segmented into non-overlapping blocks of KxK where K is target_block_size 
            ctx.block_size = target_block_size; 
            ctx.segment_h = target_block_size; 
            ctx.segment_w = target_block_size;

            // Kernel to be padded to K+K-1
            int required_block_h = ctx.segment_h + ctx.segment_h - 1;
            int required_block_w = ctx.segment_w + ctx.segment_w - 1;

            // Set FFT dims to next power of two of required block dims for efficient FFT processing
            ctx.block_h = utils::nextPowerOfTwo(required_block_h);
            ctx.block_w = utils::nextPowerOfTwo(required_block_w);
            ctx.fft_h = ctx.block_h;
            ctx.fft_w = ctx.block_w;

            std::cout << "Each block is of dimension " << "h:" << ctx.block_h  << "w:" << ctx.block_w << std::endl; 
            ctx.num_blocks_h = ceil(static_cast<float>(in_h) / ctx.segment_h);
            ctx.num_blocks_w = ceil(static_cast<float>(in_w) / ctx.segment_w);
            ctx.total_blocks = ctx.num_blocks_h * ctx.num_blocks_w;
            std::cout << "Total number of blocks: " << ctx.total_blocks << " with " << ctx.num_blocks_h << " blocks in height and " << ctx.num_blocks_w << " blocks in width" << std::endl;
            
            // Load in the test_image and test_filter into ctx, 
            std::cout << "Using provided image and filter for input..." << std::endl;
            ctx.h_input.resize(in_w * in_h);
            ctx.h_filter.resize(f_w * f_h);
            ctx.h_padded_filter.resize(ctx.block_h * ctx.block_w);

            // Convert cv::Mat to std::vector<float> for filter
            //pad mat filter to block dims then convert 
            //shift filter to center of padded block for convolution processing in frequency domain, since convolution is equivalent to cross-correlation which has the same output as convolution with a flipped kernel, so we can achieve the desired convolution output by centering the filter in the padded block and performing cross-correlation in the frequency domain, which avoids the need to perform an explicit flip of the kernel and simplifies the data preparation process for FFT-based convolution

            cv::Mat padded_filter = cv::Mat::zeros(ctx.block_h, ctx.block_w, CV_32F);
            cv::Mat test_filter_float;
            test_filter.convertTo(test_filter_float, CV_32F);
            int cx = ctx.block_w / 2;
            int cy = ctx.block_h / 2;
            // int filter_cx = f_w / 2;
            // int filter_cy = f_h / 2;
            for (int i = 0; i < f_h; ++i) {
                for (int j = 0; j < f_w; ++j) {
                    int x = (i - cy + ctx.block_h) % ctx.block_h;
                    int y = (j - cx + ctx.block_w) % ctx.block_w;
                    ctx.h_padded_filter[x * ctx.block_w + y] = test_filter_float.at<float>(i, j);
                }
            }

            // Load input 
            for (int i = 0; i < in_h; ++i) {
                for (int j = 0; j < in_w; ++j) {
                    ctx.h_input[i * in_w + j] = static_cast<float>(test_image.at<uchar>(i, j));
                }
            }
            for (int i = 0; i < f_h; ++i) {
                for (int j = 0; j < f_w; ++j) {
                    ctx.h_filter[i * f_w + j] = static_cast<float>(test_filter.at<float>(i, j));
                }
            }
        }
    }
    // Pooling Entry: 
    else{
        ctx.f_h = f_h; 
        ctx.f_w = f_w; // here f_h and f_w represent pool height and width respectively

        ctx.out_h = ((in_h - f_h + 2 * pad) / stride) + 1;
        ctx.out_w = ((in_w - f_w + 2 * pad) / stride) + 1;

        //maintain dims for simplicity 
        ctx.h_input.resize(in_w * in_h);
        //2. Calculate FFT dimensions I.e Next power of two of input image ;
        ctx.fft_h = utils::nextPowerOfTwo(in_h);
        ctx.fft_w = utils::nextPowerOfTwo(in_w);
        //3. Allocate and initialize host memory for input, filter and output

        std::cout << "Initalised context params as : in_h: " << ctx.in_h << " in_w: " << ctx.in_w << " f_h: " << ctx.f_h << " f_w: " << ctx.f_w << " stride: " << ctx.stride << " pad: " << ctx.pad << std::endl;
        std::cout << "Calculated output dimensions as out_h: " << ctx.out_h << " out_w: " << ctx.out_w << std::endl;
        if(!image_test){
            std::cout << "Generating random input and filter for pooling test..." << std::endl;
            for (int i = 0; i < in_w * in_h; ++i) {
                ctx.h_input[i] = static_cast<float>(rand() % 10); // values between 0 and 9
            }
        }
        else{
            std::cout << "Using provided image for pooling input..." << std::endl;
            for (int i = 0; i < in_h; ++i) {
                for (int j = 0; j < in_w; ++j) {
                    ctx.h_input[i * in_w + j] = static_cast<float>(test_image.at<uchar>(i, j));
                }
            }
        }

        // resize h_padded_input to fft dimensions and copy data with zero padding
        ctx.h_padded_input.resize(ctx.fft_w * ctx.fft_h, 0.0f);
        cv::Mat padded_input = cv::Mat::zeros(ctx.fft_h, ctx.fft_w, CV_32F);
        // place input image centred in padded_input for correct convolution output positioning in frequency domain
        int cx = ctx.fft_w / 2;
        int cy = ctx.fft_h / 2;
        // int input_cx = in_w / 2;
        // int input_cy = in_h / 2;
        for (int i = 0; i < in_h; ++i) {
            for (int j = 0; j < in_w; ++j) {
                int x = (i - cy + ctx.fft_h) % ctx.fft_h;
                int y = (j - cx + ctx.fft_w) % ctx.fft_w;
                ctx.h_padded_input[x * ctx.fft_w + y] = ctx.h_input[i * ctx.in_w + j];
            }
        }
    }
}

/**
    @brief GPU Memory setup for 2DConv test
     Allocates device memory for input, filter and output in both float and complex formats, and copies host data to device.
*/
template <bool isPooling>
void convolution_test::setupGPUMemory(ConvContext& ctx){
    /**
        pipeline:
        1. set device options , gpu for direct and spectral convs, cpu for torch conv
        A. Direct Conv2D Memory Setup - allocate and copy input, filter and output arrays in float format
        B. Torch Setup and Automatic padding - create torch tensors from host data, pad to FFT dimensions using torch's built in padding, and prepare for copy to device
        C. Spectral Conv2D Memory Setup - allocate complex memory for FFT-based convolution
        D. Convert Padded Tensors to Complex format and copy to device - launch kernels to convert padded float data to complex format on device (imaginary parts set to 0)

    */
    torch::Device device(torch::kCUDA, 0);
    auto cpu_options = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCPU);
    auto gpu_options = torch::TensorOptions().dtype(torch::kFloat).device(device);
    size_t free_start, free_end, total, actual_cost; 
    nvtxRangePushA("Setup_Total");

    if (!isPooling){
        
        //A. Direct Conv2D Memory Setup
        std::cout << "Direct Conv Float ptrs" << std::endl; 
        nvtxRangePushA("Setup_Direct_Malloc");
        
        // == Allocate device memory for input, filter and output in float format ==
        // == Get screenshot of inital memory state before any allocations for baseline ==
        cudaMemGetInfo(&free_start, &total);
        cudaMalloc((void**)&ctx.d_input_float, ctx.in_w * ctx.in_h * sizeof(float));
        cudaMalloc((void**)&ctx.d_filter_float, ctx.f_w * ctx.f_h * sizeof(float));
        cudaMalloc((void**)&ctx.d_output_float, ctx.out_w * ctx.out_h * sizeof(float));
        
        // DEVICE INPUT FLOAT AND FILTER FLOAT ALLOCATED, NOW COPY HOST DATA TO DEVICE
        cudaMemcpy(ctx.d_input_float, ctx.h_input.data(), ctx.in_w * ctx.in_h * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(ctx.d_filter_float, ctx.h_filter.data(), ctx.f_w * ctx.f_h * sizeof(float), cudaMemcpyHostToDevice);
        nvtxRangePop();
        // == Measure Memory reserved for direct convolution float buffers ==
        cudaMemGetInfo(&free_end, &total);
        actual_cost = free_start - free_end;
        printf("Cumulative Memory after Direct Conv2D float allocations: %zu bytes\n", actual_cost);
        cudaDeviceSynchronize();
        
        //Record Memory for Direct Conv here: 
        ctx.results["Custom_2DConv"].memory_usage_bytes = actual_cost; 
        

        // Scratch space for fft-ova-conv
        cudaMalloc((void**)&ctx.d_saved_input_float, ctx.in_w * ctx.in_h * sizeof(float));
        cudaMalloc((void**)&ctx.d_saved_filter_float, ctx.f_w * ctx.f_h * sizeof(float));
        cudaMemcpy(ctx.d_saved_input_float, ctx.d_input_float, ctx.in_w * ctx.in_h * sizeof(float), cudaMemcpyDeviceToDevice);
        cudaMemcpy(ctx.d_saved_filter_float, ctx.d_filter_float, ctx.f_w * ctx.f_h * sizeof(float), cudaMemcpyDeviceToDevice);

        if (!ctx.use_ova) {
            //B. Torch Setup and Automatic padding

            //1. Create torch tensors from host data
            nvtxRangePushA("Setup_Torch_Tensors_and_Padding");
            cudaMemGetInfo(&free_start, &total);
            auto input_raw = torch::from_blob(ctx.h_input.data(), {1, 1, ctx.in_h, ctx.in_w}, cpu_options).to(device);
            auto filter_raw = torch::from_blob(ctx.h_filter.data(), {1, 1, ctx.f_h, ctx.f_w}, cpu_options).to(device);
            auto output_raw = torch::zeros({1, 1, ctx.out_h, ctx.out_w}, gpu_options);
            ctx.t_input = input_raw.clone();
            ctx.t_filter = filter_raw.clone();
            ctx.t_output = output_raw.clone();
            nvtxRangePushA("Padding Tensors");
            
            //2. Pad tensors to FFT dimensions
            int pad_h = ctx.fft_h - ctx.in_h;
            int pad_w = ctx.fft_w - ctx.in_w;
            auto input_padded = torch::constant_pad_nd(input_raw, {0, pad_w, 0, pad_h}, 0).to(device);
            auto filter_padded = torch::constant_pad_nd(filter_raw, {0, ctx.fft_w - ctx.f_w, 0, ctx.fft_h - ctx.f_h}, 0).to(device);
            nvtxRangePop();
            nvtxRangePop();
            
            // == Measure Memory reserved for padded tensors 
            cudaMemGetInfo(&free_end, &total);
            actual_cost = free_start - free_end;
            printf("Cumulative Memory after Padded Torch Tensors: %zu bytes\n", actual_cost);
        
            nvtxRangePushA("FFTConv_Malloc_and_Copy");
            //C. Spectral Conv2D Memory Setup - Allocate complex memory for FFT-based convolution
            // Allocate complex memory for FFT-based convolution, using fft dimensions for simplicity of indexing in kernels
            cudaMemGetInfo(&free_start, &total);
            cudaMalloc((void**)&ctx.d_input_complex, ctx.fft_w * ctx.fft_h * sizeof(cuComplex));
            cudaMalloc((void**)&ctx.d_filter_complex, ctx.fft_w * ctx.fft_h * sizeof(cuComplex));
            cudaMalloc((void**)&ctx.d_output_complex, ctx.fft_w * ctx.fft_h * sizeof(cuComplex));
            cudaMalloc((void**)&ctx.d_fft_output_float, ctx.fft_w * ctx.fft_h * sizeof(float));
            nvtxRangePop();

            // == Measure Memory reserved for FFT convolution complex buffers ==
            cudaMemGetInfo(&free_end, &total);
            actual_cost = free_start - free_end;
            printf("Cumulative Memory after FFT Conv2D Complex allocations: %zu bytes\n", actual_cost);
            cudaDeviceSynchronize();

            //Record FFT_CONV2D bytes here 
            ctx.results["Spectral_Conv2D"].memory_usage_bytes = actual_cost;

            //D. Convert Padded Tensors to Complex format and copy to device
            //Extract raw pointers from padded tensors
            float* padded_input_ptr = input_padded.data_ptr<float>();
            float* padded_filter_ptr = filter_padded.data_ptr<float>();

            nvtxRangePushA("Convert_Padded_Float_to_Complex");
            //E. Convert padded float data to complex format on device (imaginary parts set to 0)
            dim3 fft_threadsPerBlock(32, 32);
            dim3 fft_blocksPerGrid((ctx.fft_w + fft_threadsPerBlock.x - 1) / fft_threadsPerBlock.x,
                                (ctx.fft_h + fft_threadsPerBlock.y - 1) / fft_threadsPerBlock.y);
            cuda_operations::float2complex<<<fft_blocksPerGrid, fft_threadsPerBlock>>>(
                ctx.fft_w, ctx.fft_h, padded_input_ptr, ctx.d_input_complex
            );
            cuda_operations::float2complex<<<fft_blocksPerGrid, fft_threadsPerBlock>>>(
                ctx.fft_w, ctx.fft_h, padded_filter_ptr, ctx.d_filter_complex
            );
            
            // Initialize output complex array to zeros
            cudaMemset(ctx.d_output_complex, 0, ctx.fft_w * ctx.fft_h * sizeof(cuComplex));
            nvtxRangePop();

            //D. save state to scratch 
            cudaMalloc((void**)&ctx.d_saved_input_complex, ctx.fft_w * ctx.fft_h * sizeof(cuComplex));
            cudaMalloc((void**)&ctx.d_saved_filter_complex, ctx.fft_w * ctx.fft_h * sizeof(cuComplex));
            cudaMemcpy(ctx.d_saved_input_complex, ctx.d_input_complex, ctx.fft_w * ctx.fft_h * sizeof(cuComplex), cudaMemcpyDeviceToDevice);
            cudaMemcpy(ctx.d_saved_filter_complex, ctx.d_filter_complex, ctx.fft_w * ctx.fft_h * sizeof(cuComplex), cudaMemcpyDeviceToDevice);
        }
        // OVA-specific memory setup 
        else {
            int in_h = ctx.in_h; int in_w = ctx.in_w;
            //int fft_h = ctx.block_h; int fft_w = ctx.block_w; // filter is padded to block dimensions for OVA convolution
            int out_h = ctx.out_h; int out_w = ctx.out_w;
            
            //B Skip tensor padding, pass input and filter as contigous cuComplex arrays 
            std::cout << "FFT-OVA-CONV cuComplex ptrs" << std::endl; 
            nvtxRangePushA("OVA_FFTConv_Malloc_and_Copy");
            
            //C Allocate complex memory for OVA convolution, using block dimensions for efficient block-wise FFT processing
            cudaMemGetInfo(&free_start, &total);
            cudaMalloc((void**)&ctx.d_input_complex, in_h * in_w * sizeof(cuComplex)); 
            cudaMalloc((void**)&ctx.d_filter_complex,  ctx.block_h *  ctx.block_w * sizeof(cuComplex)); // filter needs to fit the block_segment
            cudaMalloc((void**)&ctx.d_output_complex, out_h * out_w * sizeof(cuComplex)); // output valid conv

            cudaMalloc((void**)&ctx.workspace_block, ctx.block_h * ctx.block_w * sizeof(cuComplex)); // workspace for block-wise FFT processing
            cudaMalloc((void**)&ctx.d_fft_output_float, out_h * out_w * sizeof(float)); // to write real part of FFT output for MSE calculation and saving results, allocated with output dims since we will only copy valid conv region back to host
            cudaMalloc((void**)&ctx.d_padded_filter_OVA, ctx.block_h * ctx.block_w * sizeof(float)); // padded filter for OVA, allocated with block dims for simplicity of indexing in kernels
            
            size_t single_block_size = ctx.block_h * ctx.block_w * sizeof(cuComplex);
            for (int i =0; i < ctx.num_streams; ++i){
                cudaMalloc((void**)&ctx.d_workspaces[i], single_block_size); // workspace for storing FFTs of all blocks
                cudaMalloc((void**)&ctx.d_scratches[i], single_block_size); // scratch space for intermediate computations in OVA convolution
                cudaStreamCreate(&ctx.streams[i]);
            }
            // == Measure Memory reserved for OVA convolution complex buffers ==
            cudaMemGetInfo(&free_end, &total);
            actual_cost = free_start - free_end;
            printf("Cumulative Memory after OVA FFT Conv2D Complex allocations: %zu bytes\n", actual_cost);
            cudaDeviceSynchronize();
            
            //Record FFT_OVA_CONV2D bytes here 
            ctx.results["FFT_OVA_Conv2D"].memory_usage_bytes = actual_cost;
            
            // copy data from h_padded_filter to d_padded_filter_OVA for OVA convolution, since we need the padded filter in float format for the conversion kernel before FFT processing in OVA
            cudaMemcpy(ctx.d_padded_filter_OVA, ctx.h_padded_filter.data(), ctx.block_h * ctx.block_w * sizeof(float), cudaMemcpyHostToDevice); 

            //D Map device_input, and filter to Complex 
            dim3 convert_InputFloat_To_Complex_threadsPerBlock(16, 16);
            dim3 convert_InputFloat_To_Complex_blocksPerGrid((in_w + convert_InputFloat_To_Complex_threadsPerBlock.x - 1) / convert_InputFloat_To_Complex_threadsPerBlock.x,
                                                            (in_h + convert_InputFloat_To_Complex_threadsPerBlock.y - 1) / convert_InputFloat_To_Complex_threadsPerBlock.y);
            cuda_operations::float2complex<<<convert_InputFloat_To_Complex_blocksPerGrid, convert_InputFloat_To_Complex_threadsPerBlock>>>(
                in_w, in_h, ctx.d_saved_input_float, ctx.d_input_complex
            );
            // For filter, we need to convert the padded filter to complex format and copy to device
            dim3 convert_FilterFloat_To_Complex_threadsPerBlock(16, 16);
            dim3 convert_FilterFloat_To_Complex_blocksPerGrid((ctx.block_w + convert_FilterFloat_To_Complex_threadsPerBlock.x - 1) / convert_FilterFloat_To_Complex_threadsPerBlock.x,
                                                            (ctx.block_h + convert_FilterFloat_To_Complex_threadsPerBlock.y - 1) / convert_FilterFloat_To_Complex_threadsPerBlock.y);
            cuda_operations::float2complex<<<convert_FilterFloat_To_Complex_blocksPerGrid, convert_FilterFloat_To_Complex_threadsPerBlock>>>(
                ctx.block_w, ctx.block_h, ctx.d_padded_filter_OVA, ctx.d_filter_complex
            );

            // Initialize output complex array to zeros
            cudaMemset(ctx.d_output_complex, 0, out_h * out_w * sizeof(cuComplex));
            nvtxRangePop();
            //E. Save Scratch state for OVA convolution
            cudaMalloc((void**)&ctx.d_saved_input_complex, in_h * in_w * sizeof(cuComplex));
            cudaMalloc((void**)&ctx.d_saved_filter_complex,  ctx.block_w * ctx.block_h * sizeof(cuComplex));
            cudaMemcpy(ctx.d_saved_input_complex, ctx.d_input_complex, in_h * in_w * sizeof(cuComplex), cudaMemcpyDeviceToDevice);
            cudaMemcpy(ctx.d_saved_filter_complex, ctx.d_filter_complex, ctx.block_h * ctx.block_w * sizeof(cuComplex), cudaMemcpyDeviceToDevice);
            cudaMemcpy(ctx.d_saved_input_float, ctx.d_input_float, in_h * in_w * sizeof(float), cudaMemcpyDeviceToDevice);
            cudaMemcpy(ctx.d_saved_filter_float, ctx.d_filter_float, ctx.f_h * ctx.f_w * sizeof(float), cudaMemcpyDeviceToDevice);
        }

        //Synchronize to ensure all memory operations are complete before kernels run
        cudaDeviceSynchronize();
        nvtxRangePop();
    }
    else{
        // Pooling memory setup 
        // Standard Pool setup 
        std::cout << "Pooling Test Memory Setup" << std::endl;
        //A. Direct Pooling Memory Setup - allocate and copy input in float format
        nvtxRangePushA("Setup_Direct_Pooling_Malloc");
        // == Allocate device memory for input and output in float format ==
        cudaMemGetInfo(&free_start, &total);
        cudaMalloc((void**)&ctx.d_input_float, ctx.in_w * ctx.in_h * sizeof(float));
        cudaMalloc((void**)&ctx.d_output_float, ctx.out_w * ctx.out_h * sizeof(float));
        // DEVICE INPUT FLOAT ALLOCATED, NOW COPY HOST DATA TO DEVICE
        cudaMemcpy(ctx.d_input_float, ctx.h_input.data(), ctx.in_w * ctx.in_h * sizeof(float), cudaMemcpyHostToDevice);
        nvtxRangePop();
        // == Measure Memory reserved for direct pooling float buffers ==
        cudaMemGetInfo(&free_end, &total);
        actual_cost = free_start - free_end;
        printf("Cumulative Memory after Direct Pooling float allocations: %zu bytes\n", actual_cost);
        std::cout << "Pool mode: " << (ctx.pool_mode == 0 ? "MAX" : "AVG") << std::endl;
        // Save state to scratch for pooling test if needed in future for multiple runs with same input
        cudaMalloc((void**)&ctx.d_saved_input_float, ctx.in_w * ctx.in_h * sizeof(float));
        cudaMemcpy(ctx.d_saved_input_float, ctx.d_input_float, ctx.in_w * ctx.in_h * sizeof(float), cudaMemcpyDeviceToDevice);
    }
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
    cv::Mat test_filter,
    // result file names for runtime and mse csvs, and conv output image
    const std::string& runtime_path, 
    const std::string& mse_path, 
    const std::string& conv_output_path
)
{
    // setup paths for writing to 
    const std::string& full_path = mse_path; // TODO: this is a bit hacky, refactor later so that we pass a dedicated path for input images
    std::filesystem::path base_path(full_path);

    // Get directory path build/xxx/
    std::filesystem::path dir = base_path.parent_path();
    std::filesystem::path images_dir = dir / "results/images"; 
   
    std::string Direct_Conv_Filename = (images_dir / ("DirectConv2D_"+ std::to_string(test_input_height) + "x" + std::to_string(test_filter_height) + "x" + std::to_string(test_filter_width) + ".png")).string();
    std::string FFT_Conv_Filename = (images_dir / ("FFTConv2D_" + std::to_string(test_input_height) + "x" + std::to_string(test_filter_height) + "x" + std::to_string(test_filter_width) + ".png")).string();
    std::string OVA_Conv_Filename = (images_dir / ("FFT_OVA_Conv2D_" + std::to_string(test_input_height) + "x" + std::to_string(test_filter_height) + "x" + std::to_string(test_filter_width) + ".png")).string();

    //@NOTE: print out test parameters for debugging and verification
    std::cout << "Test Parameters:" << std::endl;
    std::cout << "Input Dimension: " << (test_input_height*test_input_width) << " (HxW: " << test_input_height << "x" << test_input_width << ")" << std::endl;
    std::cout << "Filter Dimension: " << (test_filter_height*test_filter_width) << " (HxW: " << test_filter_height << "x" << test_filter_width << ")" << std::endl;
    std::cout << "Stride: " << test_stride << " Padding: " << test_padding << std::endl;
    

    std::cout << "Running 2D Convolution Test..." << std::endl;
    //1. Initalise context with provided dimensions and data
    ConvContext ctx;
    ctx.is_pooling = false;
    //OVA must be large enough to fit f_area but smaller than input area (already a clause in bench mark if f_area >= input_area continue)
    ctx.target_ova_size = std::min(((test_filter_height)), (test_input_width)); 
    std::cout << "Target OVA size set to: " << ctx.target_ova_size << " (must be >= filter area and < input area)" << std::endl;
    //1. Setup context for non_ova test
    ctx.use_ova = false;
    convolution_test::initaliseContext<false>(
        ctx, test_input_height, test_input_width, 
        test_filter_height, test_filter_width, test_stride, 
        test_padding, image_test, test_image, test_filter,
        ctx.target_ova_size, ctx.use_ova
    );

    //2. Setup GPU memory (NON-OVA)
    std::cout << "Setting up GPU memory and copying data to device for non-OVA test..." << std::endl;
    convolution_test::setupGPUMemory<false>(ctx);

    //3. Run Conv tests: FFT_Conv, Direct Conv
    std::cout << "Running Direct Convolution..." << std::endl;
    convolution_test::run_direct(ctx);
    cudaDeviceSynchronize(); // ensure kernels are finished before starting next test for accurate timing and fair MSE comparison
    std::cout << "Running Spectral Convolution..." << std::endl;
    convolution_test::run_FFTConv(ctx);
    cudaDeviceSynchronize();
    std::cout << "2D Convolution test executed." << std::endl;
    //4. Write to CSV and output images for non-OVA test
    float mse_ = utils::MeasureError(ctx.results["Custom_2DConv"].data, ctx.results["Spectral_Conv2D"].data);
    convolution_test::mse(ctx, image_test, mse_path, mse_);
    convolution_test::runtime(ctx, image_test, runtime_path, ctx.results["Custom_2DConv"].time_ms, ctx.results["Spectral_Conv2D"].time_ms);
    
    //6. Save output images for visual comparison if this is an image test
    if(image_test){
        std::cout << "Saving output images for non-OVA test..." << std::endl;
        utils::saveOutputImage(Direct_Conv_Filename, ctx.results["Custom_2DConv"].data, ctx.out_w, ctx.out_h);
        utils::saveOutputImage(FFT_Conv_Filename, ctx.results["Spectral_Conv2D"].data, ctx.out_w, ctx.out_h);
    }


    //5. Reset context and GPU memory for OVA test
    std::vector<float> ground_truth_data = ctx.results["Custom_2DConv"].data;
    float direct_time_ms = ctx.results["Custom_2DConv"].time_ms;
    float spectral_time_ms = ctx.results["Spectral_Conv2D"].time_ms;
    size_t direct_conv_bytes = ctx.results["Custom_2DConv"].memory_usage_bytes;
    size_t FFT_Conv2D_bytes = ctx.results["Spectral_Conv2D"].memory_usage_bytes; 
    convolution_test::freeContext(ctx); // free all GPU memory and reset context before next test

    std::cout << "Starting FFT-OVA Convolution test...";
    std::cout << "Target OVA size: " << ctx.target_ova_size << std::endl;
    ctx.use_ova = true; // set flag for OVA test

    //6. Setup context and GPU memory for OVA test, then run OVA convolution and write results to CSV and output image
    convolution_test::initaliseContext<false>(
        ctx, test_input_height, test_input_width, 
        test_filter_height, test_filter_width, test_stride, 
        test_padding, image_test, test_image, test_filter,
        ctx.target_ova_size, ctx.use_ova
    );
    std::cout << "Setting up GPU memory and copying data to device for OVA test..." << std::endl;
    convolution_test::setupGPUMemory<false>(ctx);
    //early exit if OVA_block_dim exceeds input dim 
    if ((ctx.block_w) >= (test_input_width) || (ctx.block_h) >= (test_input_height)){
        std::cout << "Block dims: " << std::to_string(ctx.block_h) << "x" << std::to_string(ctx.block_w) << "Input Image Dims: " << std::to_string(test_input_height) << "x" << std::to_string(test_input_width) << ", skipping OVA test as it would not be valid..." << std::endl;
        convolution_test::freeContext(ctx);
        return;
    }
    std::cout << "Running FFT-OVA Convolution..." << std::endl;
    convolution_test::run_FFTConv(ctx); // TODO: replace with dedicated OVA Function, this may be confusing mb
    cudaDeviceSynchronize();
    std::cout << "OVA Convolution test executed." << std::endl;
    float mse_ova = utils::MeasureError(ground_truth_data, ctx.results["FFT_OVA_Conv2D"].data);
    convolution_test::mse(ctx, image_test, mse_path, mse_ova);
    convolution_test::runtime(ctx, image_test, runtime_path, direct_time_ms, ctx.results["FFT_OVA_Conv2D"].time_ms);
    
    //7. Save output image for OVA test if this is an image test
    if (image_test){
        std::cout << "Saving output image for OVA test..." << std::endl;
        utils::saveOutputImage(OVA_Conv_Filename, ctx.results["FFT_OVA_Conv2D"].data, ctx.out_w, ctx.out_h);
    }

    //8. Record the 3 algorithms memory usage
    convolution_test::memory_usage(
        ctx,
        "Memory_Usage_File",
        direct_conv_bytes, 
        FFT_Conv2D_bytes,
        ctx.results["FFT_OVA_Conv2D"].memory_usage_bytes
    );

    //9. Free all GPU memory and reset context
    convolution_test::freeContext(ctx);
    
}

void convolution_test::run_benchmarks(){
    //root /WebCNN/c++/cuda/build
    //data /WebCNN/data/userdata/
    //resized image /WebCNN/data/userdata/resized_images/
    //measurements /WebCNN/c++/cuda/build/results/convolution/
    std::filesystem::path project_root = std::filesystem::current_path().parent_path().parent_path().parent_path();

    //1. Define image list dims, 
    std::vector<convolution_test::ConvConfig> image_test_dims = {
        {"64x64", 64},
        {"128x128", 128},
        {"256x256", 256},
        {"512x512", 512},
        {"1024x1024", 1024},
        {"2048x2048", 2048},
    };

    //2. Define filter dims
    std::vector<convolution_test::ConvConfig> filter_test_dims = {
        {"3x3", 3},
        {"5x5", 5},
        {"11x11", 11},
        {"16x16", 16},
        {"32x32", 32},
        {"64x64", 64},
        {"128x128", 128},
        {"256x256", 256},
        {"512x512", 512},
        {"1024x1024", 1024},
        // {"2048x2048", 2048},
    }; 

    //3. Loop through images in data/userdata/resized_images/ and run convolution tests with each filter, saving results to results/convolution/
    std::string data_path = project_root.string() + "/data/userdata/resized_images/";
    std::string results_path = std::filesystem::current_path().string(); // results will be saved to /WebCNN/c++/cuda/build/results/convolution/

    std::cout << "Starting convolution benchmarks on images in: " << data_path << std::endl;
    std::cout << "Results will be saved to: " << results_path + "/results/benchmarks/" << std::endl;
    using recursize_directory_iterator = std::filesystem::recursive_directory_iterator;
    for (const auto& entry : recursize_directory_iterator(data_path)) {
        if (entry.is_regular_file()) {
            std::string image_path = entry.path().string();
            std::string image_name = entry.path().stem().string();
            //Read in image 
            std::cout << "Running convolution tests for image: " << image_name << std::endl;
            cv::Mat image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
            
            if (image.empty()) {
                std::cerr << "Error: Could not load image at " << image_path << std::endl;
                continue;
            }
            
            //Properties of the current image 
            int img_w = image.cols;
            int img_h = image.rows;
            std::string dim_str = std::to_string(img_w) + "x" + std::to_string(img_h);
            std::cout << "\n--- Processing Image: " << image_name << " (" << dim_str << ") ---" << std::endl;
            
            //loop through filter dims and run tests
            for (const auto& filter_dim : filter_test_dims) {
                if (filter_dim.size >= img_w || filter_dim.size >= img_h) {
                    std::cout << "Skipping filter size " << filter_dim.size << "x" << filter_dim.size 
                              << " for image " << image_name << " due to larger dimensions than the image." << std::endl;
                    continue;
                }
                std::cout << "Testing with filter size: " << filter_dim.size << "x" << filter_dim.size << std::endl;
                
                // Create a simple averaging filter for testing
                cv::Mat filter = cv::Mat::ones(filter_dim.size, filter_dim.size, CV_32F) / (filter_dim.size * filter_dim.size);
                
                //Run Convolution test
                convolution_test::test2DConvolution<true>(
                    image.rows,
                    image.cols,
                    filter.rows,
                    filter.cols,
                    1, // stride
                    0, // padding
                    image,
                    filter,
                    /**
                        Results Structure 
                        RUNTIME: 
                        Runtime_ImageName_FilterSize_ **.csv
                        MSE:
                        MSE_ImageName_FilterSize_ ** .csv
                        CONVOLUTION:
                        CONV_METHOD_ImageName_FilterSize_ **.png
                    */
                    // Runtime file : Runtime_ImageName_FilterDim
                    "Runtime_",
                    // MSE file : ImageName_InputDim_FilterDim
                    "MSE_",
                    // Conv output image : Conv_ImageName_FilterDim
                    "ConvOutput_" + image_name  +"_" + std::to_string(filter_dim.size)+"x"+std::to_string(filter_dim.size)
                );
            }
        }
    }
}
void convolution_test::test2DPooling(
    int image_height, 
    int image_width,
    int pool_height,
    int pool_width,
    int stride, 
    int padding,
    cv::Mat test_image,
    const std::string& runtime_path,
    const std::string& mse_path,
    const std::string& pool_output_path
){
    //Initalise context 
    ConvContext ctx; 
    ctx.pool_mode = cuda_operations::POOL_MODE::AVERAGE_POOL;
    ctx.is_pooling = true;
    ctx.use_ova = false; // OVA not implemented for pooling test, set to false to avoid unnecessary memory allocations and conversions in setupGPUMemory
    std::cout << "Initializing context for pooling test with pool size: " << pool_height << "x" << pool_width << std::endl;
    convolution_test::initaliseContext<true>(
        ctx,
        image_height, image_width,
        pool_height, pool_width,
        stride, padding,
        true, // image test
        test_image,
        cv::Mat(), // no filter for pooling
        0, // target block size not needed for pooling test
        false // not using OVA for pooling test
    );
    std::cout << "Context initialized for pooling test." << std::endl;

    std::cout << "Setting up GPU memory for pooling test..." << std::endl;
    //Setup GPU memory and copy data to device
    convolution_test::setupGPUMemory<true>(ctx);

    std::cout << "Running pooling test..." << std::endl;
    //Run pooling test
    convolution_test::runStandardPooling(ctx);
    std::cout << "Pooling test executed." << std::endl;
    //copy output from device to host for MSE calculation and saving results
    ctx.results["Pooling"].data.resize(ctx.out_w * ctx.out_h);
    cudaMemcpy(ctx.results["Pooling"].data.data(), ctx.d_output_float, ctx.out_w * ctx.out_h * sizeof(float), cudaMemcpyDeviceToHost);  
    // visualise pooling output as image and save
    cv::Mat pool_output(ctx.out_h, ctx.out_w, CV_32F, ctx.results["Pooling"].data.data());
    cv::imwrite("_pooling_output.png", pool_output);
    
    //runtime 
    convolution_test::runtime(ctx, true, runtime_path, ctx.results["Pooling"].time_ms, 0.0f); // second value is placeholder since we only have one pooling method currently
    //clear context memory
    convolution_test::freeContext(ctx);
}


void convolution_test::pooling(char* argv[]){
    std::cout << "Starting pooling tests" << std::endl; 
    // Define pooling configurations
    std::vector<convolution_test::ConvConfig> pooling_configs = {
        // {"2x2", 2},
        // {"3x3", 3},
        // {"5x5", 5},
        {"11x11", 11},
        {"16x16", 16},
        {"32x32", 32},
    };
    int num_runs = 1;
    int stride = 1;
    int padding = 0;
    // Loop through pooling configurations and run tests
    for(int i = 0; i < num_runs; ++i){
        for (const auto& config : pooling_configs) {
            std::cout << "Running pooling test with pool size: " << config.name << std::endl;
            // Load image using OpenCV
            std::string data_path = std::filesystem::current_path().parent_path().parent_path().parent_path().string() + "/data/userdata/";
            std::string image_path = data_path + "resized_images/resized_cat_512.png";
            cv::Mat image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);
            if (image.empty()) {
                std::cerr << "Error: Could not load image at " << image_path << std::endl;
                continue;
            }
            // Run pooling test
            convolution_test::test2DPooling(
                image.rows,
                image.cols,
                config.size,
                config.size,
                stride,
                padding,
                image,
                // runtime file name
                std::string("pooling_runtime_results_") + config.name + ".csv",
                // mse file name
                std::string("pooling_mse_results_") + config.name + ".csv",
                // pool output image name
                "n/a.png"
            );
        }
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
    TestMode mode = parseMode(argv[2]);
    std::vector<int> input_dims = {512};
    std::vector<int> filter_dims = {64};
    int num_runs = 1;
    int stride = 1;
    int padding = 0;
    for(int i = 0; i < num_runs; ++i){
        if (mode == TestMode::Image) {
            std::cout << "Running 2D Convolution Test on Image..." << std::endl;
            // Load image using OpenCV
            std::string data_path = std::filesystem::current_path().parent_path().parent_path().parent_path().string() + "/data/userdata/";

            //std::string image_path =  data_path + "cat.png";
            std::string image_path = data_path + "resized_images/resized_cat_512.png";
            
            cv::Mat image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);

            if (image.empty()) {
                std::cerr << "Error: Could not load image at " << image_path << std::endl;
                return;
            }
            std::cout << "Image loaded with dimensions: " << image.rows << " x " << image.cols << std::endl;

            // create block avg filter of size {filter_dim} for testing 
            for (const auto& filter_dim : filter_dims) {
                if (filter_dim > image.rows || filter_dim > image.cols) {
                    std::cout << "Skipping filter size " << filter_dim << "x" << filter_dim 
                              << " for image due to larger dimensions than the image." << std::endl;
                    continue;
                }
                std::cout << "Testing with filter size: " << filter_dim << "x" << filter_dim << std::endl;
                // Create a simple averaging filter for testing
                cv::Mat filter = cv::Mat::ones(filter_dim, filter_dim, CV_32F) / (filter_dim * filter_dim);
                
                //Run 2D convolution test with the image and filter
                convolution_test::test2DConvolution<true>(
                    image.rows,
                    image.cols,
                    filter.rows,
                    filter.cols,
                    stride,
                    padding,
                    image,
                    filter,
                    // runtime file name
                    "image_convolution_runtime_results", 
                    // mse file name
                    "image_convolution_mse_results",
                    // conv output image name
                    "image_convolution_output"
                );
            }
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
                        cv::Mat(),
                        // runtime file name
                        std::string("random_convolution_runtime_results_") + ".csv",
                        // mse file name
                        std::string("random_convolution_mse_results_") + ".csv",
                        // conv output image name
                        "n/a.png"
                    );
                }
            }
        }
    }
   
    //convolution_test::test2DConvolution(128,128, 64, 64, stride, padding); // Input {3,6,11,} filter{3, 5, 7} 
    std::cout << "All convolution tests completed." << std::endl;
}