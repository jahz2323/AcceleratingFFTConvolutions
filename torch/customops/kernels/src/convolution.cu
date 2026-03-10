#include "convolution.cuh"


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
    int in_width = input.size(2);
    int in_height = input.size(3);
    int filter_width = weights.size(2);
    int filter_height = weights.size(3);
    int output_width = ((in_width - filter_width + 2 * padding) / stride) + 1;
    int output_height = ((in_height - filter_height + 2 * padding) / stride) + 1;
    torch::Device device(torch::kCUDA,0);
    torch::Tensor output = torch::zeros({output_height, output_width}, torch::device(input.device()).dtype(input.dtype()));
    if (input.device() == device) {
        std::cout<<"GPU 2D convolution launched!"<<std::endl;
        dim3 blockSize(BLOCK_SIZE, BLOCK_SIZE);
        dim3 gridSize((output_width + BLOCK_SIZE - 1) / BLOCK_SIZE, (output_height + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
        const cudaStream_t stream = at::cuda::getCurrentCUDAStream(); 
        _2DConv<<<gridSize, blockSize, 0, stream>>>(in_width, in_height, filter_width, filter_height, stride, padding,
                                                    input.data_ptr<float>(), weights.data_ptr<float>(), output.data_ptr<float>());
        
        cudaDeviceSynchronize(); // Ensure kernel execution is complete before returning the output tensor
    }
    else{
        _CPU2DConv(in_width, in_height, filter_width, filter_height, stride, padding,
                   input.data_ptr<float>(), weights.data_ptr<float>(), output.data_ptr<float>());
    }
    
    return output;
}