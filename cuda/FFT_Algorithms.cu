/**
 * @author Jahziel Belmonte
 * @brief performance characterization of FFT algorithms 
 *  implementation of a k-sparse fft algorithm based on gpusfft and cusfft
 *  testing with CUFFT, MIT-SFFT, CUSFTT, GPUSFFT 
 * @file FFT_Algorithms.cu 
*/

#include <cuda_runtime.h>
#include <cufft.h>
#include <curand.h>
#include <iostream>
#include <vector>
#include <complex>
#include <libs/includes/opencv.hpp>
#include <filesystem>

#define INPUT_SIZE_CHECK  (1 << 27) // 2^27

 /**
 * @brief Display image using OpenCV
 * @param img: pointer to cv::Mat image to display
 * @param window_name: name of the display window
 * @param desired_width: desired width of the display window
 * @param desired_height: desired height of the display window
 * @param keep_aspect_ratio: whether to keep the aspect ratio of the image
 * @return void
  */

void displayImage(cv::Mat* img, const std::string& window_name, int desired_width = 800, int desired_height = 600, bool keep_aspect_ratio = true) {
    // Create a window for display.
    cv::namedWindow(window_name, cv::WINDOW_NORMAL);

    // resize window
    if (keep_aspect_ratio) {
        int original_width = img->cols;
        int original_height = img->rows;
        float aspect_ratio = static_cast<float>(original_width) / static_cast<float>(original_height);

        if (desired_width / aspect_ratio <= desired_height) {
            desired_height = static_cast<int>(desired_width / aspect_ratio);
        } else {
            desired_width = static_cast<int>(desired_height * aspect_ratio);
        }
    }
    cv::resizeWindow(window_name, desired_width, desired_height);

    cv::imshow(window_name, *img);
    int k = cv::waitKey(0);
    if (k == 's') {
        cv::imwrite(window_name + ".png", *img);
    }
}


/**
 * @brief Secondary Permutation Function 
 * GPU Kernel to perform secondary permutation for 2D sparse FFT
 * Signal is flattened to 1D 
 * @param d_bins: pointer to device 2d bins
 * @param d_x: pointer to device input 2d signal
 * @param n: size of the input signal
 * @param B: number of bins
 * @param d_hsigma: pointer to device hash parameters
 * @param d_filter: pointer to device 2d filters
 * @param fs: filter size
 */
__global__ void PFKernel2D(float* d_bins, float* d_x, int n, int B, float* d_hsigma, float* d_filter, int fs){
    // Get thread indices
    int idx = threadIdx.x + blockIdx.x * blockDim.x;  // get global thread index
    int idy = threadIdx.y + blockIdx.y * blockDim.y;  // get global thread index
    
    /* While threads are within filter size boundary */
    if(idx < fs && idy < fs){
        int bins_index = ((idx * B + idy) % B); 
        d_bins[bins_index] += d_x[((idx*n + idy * d_hsigma[idx*n + idy]) % n)] * d_filter[idx * fs + idy];
    }
}

/**
 * @brief Primary Permutation Function
 * GPU Kernel to perform spectrum permutation for 2D sparse FFT
 * @param d_bins: pointer to device 2d bins
 * @param d_x: pointer to device input 2d signal
 * @param d_filter: pointer to device 2d ilter
 * @param N: size of the input signal
 * @param B: number of bins
 * @param d_hsigma: pointer to device hash parameters
 * @param T: number of tiles for filter vector
 * @param R: R remaining elements to be permuted into bins 
 * 
 */
__global__ void PFTkernel2D(float* d_bins, float* d_x, float* d_filter, int N, int B, float* d_hsigma, int T, int R){
    int idx = threadIdx.x + blockIdx.x * blockDim.x;  // get global thread index
    int idy = threadIdx.y + blockIdx.y * blockDim.y;  // get global thread index
    /* Apply Permutation */ 
    /*Check if within bounds*/
    if((idx < N || idx == B) && (idy < N || idy == B)){
        /*if thread is less than B start placing frequencies to bins*/
        if(idx < B && idy < B){
            for(int j =0; j<T; j++){
                for(int k =0; k<T; k++){
                    int x_index = idx + j * B; // calculate x index 
                    int y_index = idy + k * B; // calculate y index

                    /* calculate index address of hash */
                    int sample_index_x = ((x_index * d_hsigma) % N); 
                    int sample_index_y = ((y_index * d_hsigma) % N);

                    /* apply gaussian or defined filter and place value into bin */
                    d_bins[x_index * N + y_index] += d_x[((sample_index_x) * N + (sample_index_y))] * d_filter[((sample_index_x) * N + (sample_index_y))];
                }
            }
        }
        if (idx == B || idy == B){
            for (int j =0; j< T; j++){
                int x_index = idx + j * T; 
                int y_index = idy + j * T;

                d_bins[x_index * N + y_index] += d_x[((x_index * d_hsigma) % N * N + (y_index * d_hsigma) % N)] * d_filter[((x_index * d_hsigma) % N * N + (y_index * d_hsigma) % N)];
            }
        } 
    }
}

/**
 * @brief GPU Kernel to compute frequency and time components of input signal
 * preparing filter window for sublinear hashing of signal
 * Gaussian or Dolph-Chebyshev Filter used gather time and frequency components
 * G(x,y) =  1/sqrt(2pi*sigma^2) * exp(-x^2  +  y^2 / 2sigma^2)
 * G(f) =  exp(-2pi^2*sigma^2 * (f^2 + g^2))
 * @param N: size of the input signal
 */
__global__ void FilterComponentsKernel(int N, float* filter_freq, float* filter_time, float sigma, float PI = 3.14159265358979323846f){
     // Get thread indices
    int idx = threadIdx.x + blockIdx.x * blockDim.x;  // get global thread index 
    int idy = threadIdx.y + blockIdx.y * blockDim.y;  // get global thread index

    if (idx < N && idy < N){
        // Frequency component 
        float f = idx; 
        float g = idy;
        int filter_time_index = idx * N + idy;
        int filter_freq_index = idx * N + idy;
        // place into time signal vector  g(x,y)
        filter_time[filter_time_index] = (1.0f / sqrtf(2.0f * PI * sigma * sigma)) * exp(-(f*f + g*g) / 2.0f * sigma * sigma);

        // place into freq signal vectors 
        filter_freq[filter_freq_index] = exp(-2.0f * PI * PI * sigma * sigma) * (exp(-(f*f + g*g)));
    }
}
/**
 * @brief GPU Host Function to call GPU Permutation and Filtering Kernels
 */
__host__ void PermutationFilterGPU(dim3 numBlocks, dim3 threads_per_block, float* dx, int B, float* d_filter, int fs, float* d_bins, float* d_hsigma){
    /* Calculate T number of tiles for filter vector and r remaining elemnts of the filter */
    int t = fs / B; 
    int r = fs % B;
    
    /* Define GPU kernel launch parameters */

    /* GPU-SFFT if input size is < 2^27 call PFTKern ELSE call PFKern */
    if (fs <= INPUT_SIZE_CHECK) {
        PFTkernel2D<<<numBlocks, threads_per_block>>>( d_bins, dx, d_filter, fs, B, d_hsigma, t, r);

    }
    else{
        PFKernel2D<<<numBlocks, threads_per_block>>>( d_bins, dx, fs, B, d_hsigma, d_filter, fs);
    }
}

/**
 * @brief Outer Loop for 2D Sparse FFT
 * @param input_signal: pointer to host input 2d signal
 */
#define LOOP_ITERATIONS 48 // l
#define k 1024 // sparsity
#define B k // Number of bins
#define Bt 2*B // Bins threshold
#define W  32 // window size 
#define L 32 // Hashing Rounds - 
#define Lc  8 // subset of rounds to find indices of significant frequencies
#define Ll 8 // Conditional check to determine to run Reverse Hashing function

__host__ void OuterLoop2D(float* input_signal, int N){
    /* Define filter size */
    int Fs = N; // filter size
    dim3 img_size(N,N);
    dim3 threads_per_block(16,16);

    dim3 filter_blocks((Fs + threads_per_block.x - 1) / threads_per_block.x,
                        (Fs + threads_per_block.y - 1) / threads_per_block.y);
    dim3 bin_blocks((B + threads_per_block.x - 1) / threads_per_block.x,
                        (B + threads_per_block.y - 1) / threads_per_block.y);

    /* Device pointers */
     
    std::vector<float> h_input_signal(N * N, 0.0f); // host input signal flattened to 1D

    /* Create device pointer */
    float* d_input_signal = nullptr; 
    float* d_bins_time = nullptr; 
    float* d_bins_freq = nullptr;
    float* d_real_signal = nullptr;
    float* d_imag_signal = nullptr;
    float* d_Hsigma = nullptr;
    float* d_filter_freq = nullptr;
    float* d_filter_time = nullptr;
   
    /* Allocate memory on the GPU */
    cudaMalloc((void**)&d_input_signal, N * N * sizeof(float));
    cudaMemcpy(d_input_signal, h_input_signal.data(), N*N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMalloc((void**)&d_bins_time, B * B * LOOP_ITERATIONS * sizeof(float));
    cudaMalloc((void**)&d_bins_freq, B * B * LOOP_ITERATIONS * sizeof(float));
    cudaMalloc((void**)&d_real_signal, N * N * sizeof(float));
    cudaMalloc((void**)&d_imag_signal, Lc * Lc * Bt * sizeof(float));
    cudaMallocManaged((void**)&d_Hsigma, L * L * sizeof(float));

    /* Allocate space for d_filter_freq and d_filter_time */
    cudaMalloc((void**)&d_filter_freq, Fs * Fs * sizeof(float));
    cudaMalloc((void**)&d_filter_time, Fs * Fs * sizeof(float));
    
    float filter_sigma = N / (2.0f * B); // filter sigma value
    float PI = 3.14159265358979323846f;

    /* Call Filter Components Kernel */
    FilterComponentsKernel<<<filter_blocks, threads_per_block>>>(Fs, d_filter_freq, d_filter_time, filter_sigma, PI);
    for(int i =0 ; i< LOOP_ITERATIONS; i++){
        /* Generate sigma */
        int permutation_sigma = rand() % N;
        d_Hsigma[i] = modInv(permutation_sigma, N); // modular inverse of sigma
        /* Call Permutation and Filtering GPU */
        PermutationFilterGPU(bin_blocks, threads_per_block, d_input_signal, B, d_filter_freq, Fs, d_bins_time, d_Hsigma[i]);
        /* Permuation and filtering */
        // calculate time and fequency components of filter

        /* FFT and Cut-Off*/
        // if (LOOP_ITERATIONS < Ll){
        //     // Call Reverse Hash Function
        //     /* Reverse Hash Function */
        //     std::cout << "Reverse Hash Function called at iteration: " << i << std::endl;
        // }
    }
    /* Free device memory */
    cudaFree(d_input_signal);
    cudaFree(d_bins_time);
    cudaFree(d_bins_freq);
    cudaFree(d_real_signal);
    cudaFree(d_imag_signal);
    cudaFree(d_Hsigma);
    cudaFree(d_filter_freq);
    cudaFree(d_filter_time);

}


int main(){
    /* Load current dir */
    std::cout << "Current working directory: " << std::filesystem::current_path() << std::endl;
    std::string image_path = "/images/cat.jpg";
    std::string full_img_path = std::filesystem::current_path().parent_path().string() + image_path;
    std::cout << "Full image path: " << full_img_path << std::endl;
    cv::Mat img = cv::imread(full_img_path);

    /* Check if image is loaded */
    if(!img.data){
        std::cerr << "Could not read the image: " << image_path << std::endl;
        return 1;
    }

    /* Check image properties i.e shape */
    std::cout << "Image rows: " << img.rows << ", cols: " << img.cols << ", channels: " << img.channels() << std::endl;
    cv::resize(img, img, cv::Size(512, 512)); // Resize image to 512x512
    cv::cvtColor(img, img, cv::COLOR_BGR2GRAY); // Convert to grayscale

    
    // Expand input image to optimal size 
    cv::Mat padded; 
    int m = cv::getOptimalDFTSize(img.rows);
    int n = cv::getOptimalDFTSize(img.cols);
    cv::copyMakeBorder(img, padded, 0, m - img.rows, 0, n - img.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));

    /* Call Outer Loop 2D Sparse FFT */
    OuterLoop2D((float*)padded.data, padded.rows);
    /** DFT OF 2D Data */
    {
        // // Expand input image to optimal size
        // cv::Mat padded; 
        // int m = cv::getOptimalDFTSize(img.rows);
        // int n = cv::getOptimalDFTSize(img.cols);
        // cv::copyMakeBorder(img, padded, 0, m - img.rows, 0, n - img.cols, cv::BORDER_CONSTANT, cv::Scalar::all(0));

        // // Make place for both the complex and the real values
        // cv::Mat planes[] = {cv::Mat_<float>(padded), cv::Mat::zeros(padded.size(), CV_32F)}; // real and imaginary parts
        // cv::Mat complexI;
        // cv::merge(planes, 2, complexI); // Merge into a complex matrix

        // // Perform DFT
        // cv::dft(complexI, complexI);

        // // transform real and imaginary parts back to planes
        // cv::split(complexI, planes);
        // cv::magnitude(planes[0], planes[1], planes[0]); // magnitude
        // cv::Mat magI = planes[0];

        // // switch to logarithmic scale
        // magI += cv::Scalar::all(1);
        // cv::log(magI, magI);

        // // crop and rearrange 
        // magI = magI(cv::Rect(0, 0, magI.cols & -2, magI.rows & -2));

        // // rearrange the quadrants of Fourier image
        // int cx = magI.cols / 2;
        // int cy = magI.rows / 2;

        // cv::Mat q0(magI, cv::Rect(0, 0, cx, cy));   // Top-Left
        // cv::Mat q1(magI, cv::Rect(cx, 0, cx, cy));  // Top-Right
        // cv::Mat q2(magI, cv::Rect(0, cy, cx, cy));  // Bottom-Left
        // cv::Mat q3(magI, cv::Rect(cx, cy, cx, cy)); // Bottom-Right

        // cv::Mat tmp;
        // q0.copyTo(tmp);
        // q3.copyTo(q0);
        // tmp.copyTo(q3);

        // q1.copyTo(tmp);
        // q2.copyTo(q1);
        // tmp.copyTo(q2);

        // // Normalize the magnitude image for display
        // cv::normalize(magI, magI, 0, 1, cv::NORM_MINMAX);

        // // Display magnitude image
        // displayImage(&magI, "Magnitude Spectrum");
    }
    
    
    return 0;
}