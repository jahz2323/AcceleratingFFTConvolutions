#pragma once 

#include <chrono>
#include <iostream>
#include <torch/torch.h>
#include <math_constants.h>
#include <utils.cuh>
#include <cuda_operations.cuh>

using namespace cuda_operations;


// Define test modes
enum class TestMode {
    Random,
    Image,
    Unknown
};


inline TestMode parseMode(char* arg) {
    if (!arg) return TestMode::Unknown;
    std::string s(arg);
    if (s == "0" || s == "random") return TestMode::Random;
    if (s == "1" || s == "image")  return TestMode::Image;
    return TestMode::Unknown;
}


class convolution_test {
public:
    struct OVABlock{
        float* data;
    };
    struct ConvContext {
        bool use_ova;
        int target_ova_size = 128; // default block size for OVA, can be tuned based on filter and input sizes, and GPU capabilities

        int in_h, in_w, f_h, f_w, stride, pad; 
        int out_h, out_w;
        int fft_h, fft_w;

        // OVA params
        int block_size; 
        int block_h, block_w;
        int segment_h, segment_w;
        int num_blocks_h, num_blocks_w;
        int total_blocks;
        static constexpr int num_streams = 2; // for OVA convolution, can be tuned based on GPU capabilities and block count
        cudaStream_t streams[num_streams]; // array of CUDA streams for concurrent block processing in OVA convolution
        cuComplex* d_workspaces[num_streams]; // workspace for storing FFTs of blocks in OVA convolution
        cuComplex* d_scratches[num_streams];
        cuComplex* workspace_block; 

        //Host Data 
        std::vector<float> h_input, h_filter, h_output; 
        std::vector<float> h_padded_input, h_padded_filter;

        //Device Data 
        //Direct Conv
        float *d_input_float, *d_filter_float, *d_output_float;
        
        //FFT Conv
        cuComplex *d_input_complex, *d_filter_complex, *d_output_complex;
        float *d_padded_filter_OVA; // store padded filter to segment for ova;
        float *d_fft_output_float; // for storing real part of FFT output after conversion, used for MSE and saving results, allocated with fft dims for simplicity of indexing
        //Torch Data
        torch::Tensor t_input, t_filter, t_output;

        //Scratch space for save_state 
        cuComplex* d_saved_input_complex, *d_saved_filter_complex;
        float* d_saved_input_float, *d_saved_filter_float;

        //Pool mode 
        cuda_operations::POOL_MODE pool_mode;
        bool is_pooling = false; 
    

        //Results Map 
        struct Results{
            float time_ms;
            float mse;
            size_t memory_usage_bytes;
            std::vector<float> data; 
        };
        std::map<std::string, Results> results;
    };

    struct ConvConfig {
        std::string name;
        int size;
    };
    
    template<bool isPooling>
    static void initaliseContext(
        ConvContext& ctx, 
        int in_h, int in_w, 
        int f_h, int f_w, 
        int stride, int pad, 
        bool image_test, 
        cv::Mat test_image, cv::Mat test_filter, 
        int target_block_size,
        bool use_ova = false
    ); // handles image vs random data loading

    template<bool isPooling>
    static void setupGPUMemory(ConvContext& ctx); // allocates and copies data to device
    static void setupFFTContext(ConvContext& ctx); // allocates complex ptrs
    static void setupDirectContext(ConvContext& ctx); // allocates float ptrs
    static void freeContext(ConvContext& ctx); // frees all device memory

    static void run_benchmarks(); // runs benchmarks for various configurations and saves results to file, can be called from main or individual tests
    static void test2DPooling(
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
    );

    template <bool image_test> 
    static void test2DConvolution(
        int, 
        int, 
        int, 
        int, 
        int, 
        int, 
        cv::Mat, 
        cv::Mat,
        const std::string& runtime_file_name,
        const std::string& mse_file_name,
        const std::string& conv_output_name
    );
    static void test1DConvolution();
    static void testFFTConvolution();
    static void convolve(char* argv[]);
    static void pooling(char* argv[]);

    static void run_cuFFT(ConvContext& ctx);
    static void run_direct(ConvContext& ctx);
    static void run_torch(ConvContext& ctx);
    static void run_FFTConv(ConvContext& ctx);
    static void run_OptimisedFFTConv(ConvContext& ctx);
    
    //Pooling
    static void runStandardPooling(ConvContext& ctx);
    static void runSpectralPooling(ConvContext& ctx);

    static void resetFFTContext(ConvContext& ctx); // resets complex buffers to saved state for fair runtime comparisons between spectral methods
    static void runtime(
        ConvContext& ctx, 
        bool image_test,
        const std::string& full_path,
        float custom_conv2d_milliseconds, 
        float spectral_conv2d_milliseconds,
        float torch_conv2d_milliseconds = 0.0f, 
        float shared_memory_spectral_conv2d_milliseconds = 0.0f, 
        float cuFFT_conv2d_milliseconds = 0.0f
    );
    static void mse(
        ConvContext& ctx,
        bool image_test,
        const std::string& full_path,
        float mse,
        float mse_optimised = 0.0f, 
        float mse_cufft = 0.0f 
    );
    static void memory_usage(
        ConvContext& ctx,
        const std::string& full_path, 
        size_t Direct_Conv2D_memory_bytes,
        size_t FFT_Conv2D_memory_bytes,
        size_t FFT_OVA_Conv2D_memory_bytes = 0 // incase not called
    );
};