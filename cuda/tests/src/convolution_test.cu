#include "convolution_test.cuh"
#include "dataprep.hpp"
#include <nvtx3/nvToolsExt.h> // For NVTX profiling ranges

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
        "Spectral_2DFFTConv",
        std::to_string(in_height) + "x" + std::to_string(in_width),
        std::to_string(filter_height) + "x" +  std::to_string(filter_width),
        std::to_string(mse_val),
    };

    std::filesystem::path base_path(full_path);

    std::string csv_path = base_path.string() + ".csv";
     // Get directory path build/xxx/
    std::filesystem::path dir = base_path.parent_path(); 
    std::string stem = base_path.filename().string();
    std::string mse_csv_name = (dir / ("MSE_" + stem + ".csv")).string();

    // Write MSE results to CSV file, if file already exists, append new results as a new row, if not create new file and write header and results, if csv is not present, create a new file. If csv is present, check if header is present, if not write header, then append results as a new row. If csv is present and header is present, just append results as a new row.
    utils::writeCSV(mse_csv_name, mse_csv_content, mse_csv_header);

    // Save output images for visual comparison if this is an image test
    if(image_test){
        /**
            Image Saving:
             - Save output images from direct convolution and spectral convolution for visual comparison
             - Filenames should be descriptive and include test parameters for easy identification, e.g.:
        */
 
        //append method to stem and add png extension
        std::string direct_conv_filename = (dir / ("DirectConv2D_" + stem + ".png")).string();
        std::string spectral_conv_filename = (dir / ("SpectralConv2D_" + stem + ".png")).string();
            
        //Print out where images are being saved
        std::cout << "Saving images to\n: Direct Conv2D Output Image: " << direct_conv_filename << "\nSpectral Conv2D Output Image: " << spectral_conv_filename << std::endl;
        
        // Save direct convolution output image
        utils::saveOutputImage(direct_conv_filename, ctx.results["Custom_2DConv"].data, ctx.out_w, ctx.out_h);
        // Save spectral convolution output image
        utils::saveOutputImage(spectral_conv_filename, ctx.results["Spectral_Conv2D"].data, ctx.out_w, ctx.out_h);
    }
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
        "Spectral_Conv2D",
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
    //Free device memory
    cudaFree(ctx.d_input_float);
    cudaFree(ctx.d_filter_float);
    cudaFree(ctx.d_output_float);
    cudaFree(ctx.d_input_complex);
    cudaFree(ctx.d_filter_complex);
    cudaFree(ctx.d_output_complex);
    cudaFree(ctx.d_fft_output_float);
    cudaFree(ctx.d_saved_filter_complex);
    cudaFree(ctx.d_saved_input_complex);
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
    nvtxRangePop();

    std::vector<float> direct_conv_output(ctx.out_w * ctx.out_h, 0.0f);
    cudaMemcpy(direct_conv_output.data(), ctx.d_output_float, ctx.out_w * ctx.out_h * sizeof(float), cudaMemcpyDeviceToHost);
    //save to ctx.results for MSE calculation and image saving
    ctx.results["Custom_2DConv"].data = direct_conv_output;

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
    std::cout << "-------- Running Custom FFT Convolution Kernel --------" << std::endl;
    cudaEvent_t start, stop;
    cudaEventCreate(&start); cudaEventCreate(&stop);

    cudaEventRecord(start);
    // RUN KERNEL
    nvtxRangePushA("Spectral_Conv2D_Runner");
    cuda_operations::_2D_FFTConv(
        ctx.fft_h, ctx.fft_w, ctx.fft_h, ctx.fft_w,
        ctx.d_input_complex, ctx.d_filter_complex, ctx.d_output_complex
    );
    nvtxRangePop();
    
    //Store in ctx.results["Spectral_Conv2D"].data 
    std::vector<cuComplex> fft_conv_output_complex(ctx.fft_w * ctx.fft_h);
    cudaMemcpy(fft_conv_output_complex.data(), ctx.d_output_complex, ctx.fft_w * ctx.fft_h * sizeof(cuComplex), cudaMemcpyDeviceToHost);
    dim3 convert_threadsPerBlock(16, 16);
    dim3 convert_blocksPerGrid((ctx.fft_w + convert_threadsPerBlock.x - 1) / convert_threadsPerBlock.x,
                               (ctx.fft_h + convert_threadsPerBlock.y - 1) / convert_threadsPerBlock.y);
    //convert FFT output from complex to real and copy back to host
    cuda_operations::complex2float<<<convert_blocksPerGrid, convert_threadsPerBlock>>>(
        ctx.fft_w, ctx.fft_h, ctx.d_output_complex, ctx.d_fft_output_float
    );
    cudaDeviceSynchronize();
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

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ctx.results["Spectral_Conv2D"].time_ms, start, stop);
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

void convolution_test::resetFFTContext(ConvContext& ctx){
    // Utility function to reset complex buffers to saved state for fair runtime comparisons between spectral methods
    cudaMemcpy(ctx.d_input_complex, ctx.d_saved_input_complex, ctx.fft_w * ctx.fft_h * sizeof(cuComplex), cudaMemcpyDeviceToDevice);
    cudaMemcpy(ctx.d_filter_complex, ctx.d_saved_filter_complex, ctx.fft_w * ctx.fft_h * sizeof(cuComplex), cudaMemcpyDeviceToDevice);
}
/**
    @brief Initalisation func for 2DConv 

*/
void convolution_test::initaliseContext(
    ConvContext& ctx, int in_h, int in_w, int f_h, int f_w, int stride, int pad,
    bool image_test, cv::Mat test_image, cv::Mat test_filter
){
    //1. Set dimensions and calculate derived dimensions
    ctx.in_h = in_h; ctx.in_w = in_w; ctx.f_h = f_h; ctx.f_w = f_w; 
    ctx.stride = stride; ctx.pad = pad;
    ctx.out_h = ((in_h - f_h + 2 * pad) / stride) + 1;
    ctx.out_w = ((in_w - f_w + 2 * pad) / stride) + 1;

    //2. Calculate FFT dimensions
    int target_dimensions_height = in_h + f_h - 1;
    int target_dimensions_width = in_w + f_w - 1;
    ctx.fft_h = utils::nextPowerOfTwo(target_dimensions_height);
    ctx.fft_w = utils::nextPowerOfTwo(target_dimensions_width);

    //3. Allocate and initialize host memory for input, filter and output
    ctx.h_input.resize(in_w * in_h);
    ctx.h_filter.resize(f_w * f_h);
  
    //4. Load data into host vectors - either random or from provided images
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

/**
    @brief GPU Memory setup for 2DConv test
     Allocates device memory for input, filter and output in both float and complex formats, and copies host data to device.
*/
void convolution_test::setupGPUMemory(ConvContext& ctx){
    /**
        pipeline:
        1. set device options , gpu for direct and spectral convs, cpu for torch conv
        A. Direct Conv2D Memory Setup - allocate and copy input, filter and output arrays in float format
        B. Torch Setup and Automatic padding - create torch tensors from host data, pad to FFT dimensions using torch's built in padding, and prepare for copy to device
        C. Spectral Conv2D Memory Setup - allocate complex memory for FFT-based convolution
        D. Convert Padded Tensors to Complex format and copy to device - launch kernels to convert padded float data to complex format on device (imaginary parts set to 0)

    */
    nvtxRangePushA("Setup_Total");

    torch::Device device(torch::kCUDA, 0);
    auto cpu_options = torch::TensorOptions().dtype(torch::kFloat).device(torch::kCPU);
    auto gpu_options = torch::TensorOptions().dtype(torch::kFloat).device(device);

    //A. Direct Conv2D Memory Setup
    nvtxRangePushA("Setup_Direct_Malloc");
    cudaMalloc((void**)&ctx.d_input_float, ctx.in_w * ctx.in_h * sizeof(float));
    cudaMalloc((void**)&ctx.d_filter_float, ctx.f_w * ctx.f_h * sizeof(float));
    cudaMalloc((void**)&ctx.d_output_float, ctx.out_w * ctx.out_h * sizeof(float));
    cudaMemcpy(ctx.d_input_float, ctx.h_input.data(), ctx.in_w * ctx.in_h * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(ctx.d_filter_float, ctx.h_filter.data(), ctx.f_w * ctx.f_h * sizeof(float), cudaMemcpyHostToDevice);
    nvtxRangePop();

    //B. Torch Setup and Automatic padding 
    //1. Create torch tensors from host data
    nvtxRangePushA("Setup_Torch_Tensors_and_Padding");
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

    nvtxRangePushA("FFTConv_Malloc_and_Copy");
    //C. Spectral Conv2D Memory Setup - Allocate complex memory for FFT-based convolution
    cudaMalloc((void**)&ctx.d_input_complex, ctx.fft_w * ctx.fft_h * sizeof(cuComplex));
    cudaMalloc((void**)&ctx.d_filter_complex, ctx.fft_w * ctx.fft_h * sizeof(cuComplex));
    cudaMalloc((void**)&ctx.d_output_complex, ctx.fft_w * ctx.fft_h * sizeof(cuComplex));
    
    // d_fft_output_float - to write real part of FFT output for MSE calculation and saving results, allocated with fft dims for simplicity of indexing
    cudaMalloc((void**)&ctx.d_fft_output_float, ctx.fft_w * ctx.fft_h * sizeof(float));
    nvtxRangePop();

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

    //Synchronize to ensure all memory operations are complete before kernels run
    cudaDeviceSynchronize();
    nvtxRangePop();
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
    std::cout << "Running 2D Convolution Test..." << std::endl;
    //1. Initalise context with provided dimensions and data
    ConvContext ctx;
    convolution_test::initaliseContext(ctx, test_input_height, test_input_width, test_filter_height, test_filter_width, test_stride, test_padding, image_test, test_image, test_filter);

    //2. Setup GPU memory and copy data to device
    convolution_test::setupGPUMemory(ctx);

    //3. Use data from CTX and run each convolution variant, storing results in CTX.results
    convolution_test::run_direct(ctx);
    convolution_test::run_FFTConv(ctx);
    std::cout << "2D Convolution test executed." << std::endl;

    //Reset Context 
    convolution_test::resetFFTContext(ctx);

    float mse_spectral = utils::MeasureError(ctx.results["Custom_2DConv"].data, ctx.results["Spectral_Conv2D"].data);

    //4. Save results - MSE and runtime to CSV files, and output images for visual comparison if this is an image test
    convolution_test::mse(ctx, image_test, mse_path, mse_spectral);
    convolution_test::runtime(ctx, image_test, runtime_path, ctx.results["Custom_2DConv"].time_ms, ctx.results["Spectral_Conv2D"].time_ms);
    convolution_test::freeContext(ctx); //MUST BE AT THE END
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
        {"2048x2048", 2048},
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
                if (filter_dim.size > img_w || filter_dim.size > img_h) {
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
                    results_path + "/results/benchmarks/" + image_name + "_filter_" + filter_dim.name,
                    results_path + "/results/benchmarks/" + image_name + "_filter_" + filter_dim.name,
                    results_path + "/results/images/" + image_name + "_filter_" + filter_dim.name
                );
            }
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
    
    std::vector<int> input_dims = {256};
    std::vector<int> filter_dims = {3, 5, 11 ,16, 32,64, 128, 256};
    int num_runs = 1;
    int stride = 1;
    int padding = 0;
    for(int i = 0; i < num_runs; ++i){
        if (mode == TestMode::Image) {
            std::cout << "Running 2D Convolution Test on Image..." << std::endl;
            // Load image using OpenCV
            std::string data_path = std::filesystem::current_path().parent_path().parent_path().parent_path().string() + "/data/userdata/";

            //std::string image_path =  data_path + "cat.png";
            std::string image_path = data_path + "resized_images/";
            
            cv::Mat image = cv::imread(image_path, cv::IMREAD_GRAYSCALE);

            if (image.empty()) {
                std::cerr << "Error: Could not load image at " << image_path << std::endl;
                return;
            }
            std::cout << "Image loaded with dimensions: " << image.rows << " x " << image.cols << std::endl;

            // block avg 
            cv::Mat filter = (cv::Mat_<float>(3,3) <<
                1.0/9.0, 1.0/9.0, 1.0/9.0,
                1.0/9.0, 1.0/9.0, 1.0/9.0,
                1.0/9.0, 1.0/9.0, 1.0/9.0
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
                filter,
                // runtime file name
                "image_convolution_runtime_results.csv", 
                // mse file name
                "image_convolution_mse_results.csv",
                // conv output image name
                "image_convolution_output.png"
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