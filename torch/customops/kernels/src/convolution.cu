#include "convolution.cuh"

__global__ void _2DConv4D(
    int64_t InChannels, int64_t OutChannels, 
    int in_width, int in_height, 
    int filter_width, int filter_height, 
    int8_t stride, int8_t padding,
    void* input, void* filters, void* output
)                   
{
    int N = blockIdx.z / InChannels; // Batch index
    int C_in = blockIdx.z % InChannels; // Input channel index
    auto idx = blockIdx.x * blockDim.x + threadIdx.x; // Output width index
    auto idy = blockIdx.y * blockDim.y + threadIdx.y; // Output height index
    int output_width = ((in_width - filter_width + 2 * padding) / stride) + 1;
    int output_height = ((in_height - filter_height + 2 * padding) / stride) + 1;
    // bounds check
    if  (idx >= output_width || idy >= output_height) return;

    float pvalue = 0.0f; 

    //Loop through input channels and accumulate results for each output channel
    for (int C_out = 0; C_out < OutChannels; C_out++){
        float* input_ptr = static_cast<float*>(input) + N * InChannels * in_width * in_height + C_in * in_width * in_height;
        float* filter_ptr = static_cast<float*>(filters) + C_out * InChannels * filter_width * filter_height + C_in * filter_width * filter_height;
        
        //Loop through filter
        for (int k = 0; k < filter_height; k++) {
            for (int l = 0; l < filter_width; l++) {
                int n_start_pos = idy * stride + k - padding;
                int m_start_pos = idx * stride + l - padding;
                if (n_start_pos >= 0 && n_start_pos < in_height &&
                    m_start_pos >= 0 && m_start_pos < in_width) {
                    pvalue += input_ptr[n_start_pos * in_width + m_start_pos] * filter_ptr[k * filter_width + l];
                }
            }
        }
    }
    int out_idx = N * OutChannels * output_width * output_height + (blockIdx.z / InChannels) * output_width * output_height + idy * output_width + idx;
    static_cast<float*>(output)[out_idx] = pvalue;
}

__global__ void _2DConv4DBackward(
    int64_t InChannels, int64_t OutChannels, 
    int in_width, int in_height, 
    int filter_width, int filter_height, 
    int8_t stride, int8_t padding,
    void* input, void* filters, 
    void* grad_output, 
    void* grad_input, 
    void* grad_filters
)                   
{
    // block computes grad for output channel and pixel 
    int batch = blockIdx.z; 
    int C_out = blockIdx.y;
    int out_x = blockIdx.x * blockDim.x + threadIdx.x;

    int out_w = ((in_width - filter_width + 2 * padding) / stride) + 1;
    int out_h = ((in_height - filter_height + 2 * padding) / stride) + 1;
    int out_area = out_w * out_h;
    if (out_x >= out_w || blockIdx.x * blockDim.x + threadIdx.x >= out_w) return;
    if (C_out >= OutChannels) return;

    int oy = out_x / out_w;
    int ox = out_x % out_w;

    // get upstream gradient 
    float dz = static_cast<float*>(grad_output)[batch * OutChannels * out_area + C_out * out_area + oy * out_w + ox];

    //loop through input channels to compute grad for filters and input
    for (int C_in = 0; C_in < InChannels; C_in++){
        float* input_ptr = static_cast<float*>(input) + batch * InChannels * in_width * in_height + C_in * in_width * in_height;
        float* filter_ptr = static_cast<float*>(filters) + C_out * InChannels * filter_width * filter_height + C_in * filter_width * filter_height;
        float* grad_input_ptr = static_cast<float*>(grad_input) + batch * InChannels * in_width * in_height + C_in * in_width * in_height;
        float* grad_filter_ptr = static_cast<float*>(grad_filters) + C_out * InChannels * filter_width * filter_height + C_in * filter_width * filter_height;
        for (int k = 0; k < filter_height; k++) {
            for (int l = 0; l < filter_width; l++) {
                int n_start_pos = oy * stride + k - padding;
                int m_start_pos = ox * stride + l - padding;
                if (n_start_pos >= 0 && n_start_pos < in_height &&
                    m_start_pos >= 0 && m_start_pos < in_width) {
                    // grad for filters is input * upstream grad
                    atomicAdd(&grad_filter_ptr[k * filter_width + l], input_ptr[n_start_pos * in_width + m_start_pos] * dz);
                    // grad for input is filter value * upstream grad
                    atomicAdd(&grad_input_ptr[n_start_pos * in_width + m_start_pos], filter_ptr[k * filter_width + l] * dz);
                }
            }
        }
    }
}

void _2DConv4DBackwardCPU(
    int64_t batch_size,
    int64_t InChannels, int64_t OutChannels, 
    int in_width, int in_height, 
    int filter_width, int filter_height, 
    int8_t stride, int8_t padding,
    void* input, void* filters, 
    void* grad_output, 
    void* grad_input, 
    void* grad_filters
){
    float* in_p = static_cast<float*>(input);
    float* filt_p = static_cast<float*>(filters);
    float* g_out_p = static_cast<float*>(grad_output);
    float* g_in_p = static_cast<float*>(grad_input);
    float* g_filt_p = static_cast<float*>(grad_filters);

    int out_w = ((in_width - filter_width + 2 * padding) / stride) + 1;
    int out_h = ((in_height - filter_height + 2 * padding) / stride) + 1;
    int out_area = out_w * out_h;
    for (int batch = 0; batch < batch_size; ++batch){
        for (int64_t j =0; j < OutChannels; ++j){ // loop over output channels
            for (int64_t i =0; i < InChannels; ++i){ // loop over input channels

                //Loop over output Grid 
                for ( int x = 0; x < out_w; ++x){
                    for (int y = 0; y < out_h; ++y){

                        // top left corner of the filter on the input
                        int x_start = x * stride - padding;
                        int y_start = y * stride - padding;

                        //upstream gradient 
                        float dz = g_out_p[batch * OutChannels * out_area + j * out_area + y * out_w + x];

                        //Loop over kernel 
                        for (int kx= 0; kx < filter_width; ++kx){
                            for (int ky = 0; ky < filter_height; ++ky){
                                int x_in = x_start + kx;
                                int y_in = y_start + ky;

                                // Check for valid input coordinates (handle padding)
                                if (x_in >= 0 && x_in < in_width && y_in >= 0 && y_in < in_height){
                                    // Update gradients
                                    g_filt_p[j * InChannels * filter_width * filter_height + i * filter_width * filter_height + kx * filter_width + ky] += in_p[batch * InChannels * in_width * in_height + i * in_width * in_height + y_in * in_width + x_in] * dz; // DL/DW += X * DL/DYpred
                                    g_in_p[batch * InChannels * in_width * in_height + i * in_width * in_height + y_in * in_width + x_in] += filt_p[j * InChannels * filter_width * filter_height + i * filter_width * filter_height + kx * filter_width + ky] * dz; // DL/DX += W * DL/Ypred
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}



__global__ void _2DConv(int in_width, int in_height, int filter_width, int filter_height, int8_t stride, int8_t padding,
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

void _CPU2DConv(int in_width, int in_height, int filter_width, int filter_height, int8_t stride, int8_t padding,
                           void* input, void* filters, void* output)                   
{
    int output_width = ((in_width - filter_width + 2 * padding) / stride) + 1;
    int output_height = ((in_height - filter_height + 2 * padding) / stride) + 1;
    for (int idy = 0; idy < output_height; idy++) {
        for (int idx = 0; idx < output_width; idx++) {
            float pvalue = 0.0f;
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
    }
}

static const int BLOCK_SIZE = 16; 
torch::Tensor _2DConvLauncher(torch::Tensor input, torch::Tensor weights, int8_t stride, int8_t padding){
    int64_t batch_size = input.size(0);
    int64_t in_channels = input.size(1);
    int64_t out_channels = weights.size(0);
    int64_t in_width = input.size(2);
    int in_height = input.size(3);
    int filter_width = weights.size(2);
    int filter_height = weights.size(3);
    int output_width = ((in_width - filter_width + 2 * padding) / stride) + 1;
    int output_height = ((in_height - filter_height + 2 * padding) / stride) + 1;
    torch::Device device(torch::kCUDA,0);
    torch::Tensor output = torch::zeros({batch_size, out_channels, output_height, output_width}, torch::device(input.device()).dtype(input.dtype()));
    if (input.device() == device) {
        std::cout<<"GPU 2D convolution launched!"<<std::endl;
        dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
        dim3 gridSize(
            (output_width + BLOCK_SIZE - 1) / BLOCK_SIZE, 
            (output_height + BLOCK_SIZE - 1) / BLOCK_SIZE,
            batch_size * in_channels // Each block processes one output pixel for a specific input channel and batch item
        );
    
        const cudaStream_t stream = at::cuda::getCurrentCUDAStream(); 
        // _2DConv<<<gridSize, blockSize, 0, stream>>>(in_width, in_height, filter_width, filter_height, stride, padding,
        //                                             input.data_ptr<float>(), weights.data_ptr<float>(), output.data_ptr<float>());
        _2DConv4D<<<gridSize, blockSize, 0, stream>>>(input.size(1), weights.size(0), in_width, in_height, filter_width, filter_height, stride, padding,
                                                    input.data_ptr<float>(), weights.data_ptr<float>(), output.data_ptr<float>());
        cudaDeviceSynchronize(); // Ensure kernel execution is complete before returning the output tensor
    }
    else{
        // _CPU2DConv(in_width, in_height, filter_width, filter_height, stride, padding,
        //            input.data_ptr<float>(), weights.data_ptr<float>(), output.data_ptr<float>());
        std::cout<<"NO DEVICE FOUND" << std::endl;
    }
    
    return output;
}

std::tuple<torch::Tensor, torch::Tensor> _2DConvBackwardLauncher(torch::Tensor grad_output, torch::Tensor input, torch::Tensor filters, int8_t stride, int8_t padding) {
    int64_t batch_size = input.size(0);
    int64_t in_channels = input.size(1);
    int64_t out_channels = filters.size(0);
    int64_t in_width = input.size(2);
    int in_height = input.size(3);
    int filter_width = filters.size(2);
    int filter_height = filters.size(3);
    int output_width = ((in_width - filter_width + 2 * padding) / stride) + 1;
    int output_height = ((in_height - filter_height + 2 * padding) / stride) + 1;

    torch::Device device(torch::kCUDA,0); 
    torch::Tensor grad_input = torch::zeros_like(input);
    torch::Tensor grad_filters = torch::zeros_like(filters); 
    if (input.device() == device){
        std::cout << "GPU Backpass called" << std::endl; 
        dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
        dim3 gridSize(
            (output_width + BLOCK_SIZE - 1) / BLOCK_SIZE, 
            (output_height + BLOCK_SIZE - 1) / BLOCK_SIZE,
            batch_size * in_channels 
        );
        const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
        _2DConv4DBackward<<<gridSize, blockSize, 0, stream>>>(input.size(1), filters.size(0), in_width, in_height, filter_width, filter_height, stride, padding,
                                                            input.data_ptr<float>(), filters.data_ptr<float>(), grad_output.data_ptr<float>(), grad_input.data_ptr<float>(), grad_filters.data_ptr<float>());
        cudaDeviceSynchronize();
    }
    else{
        std::cout << "CPU Backpass called" << std::endl;
        _2DConv4DBackwardCPU(batch_size, input.size(1), filters.size(0), in_width, in_height, filter_width, filter_height, stride, padding,
                            input.data_ptr<float>(), filters.data_ptr<float>(), grad_output.data_ptr<float>(), grad_input.data_ptr<float>(), grad_filters.data_ptr<float>());
    }
    return {grad_input, grad_filters};
}