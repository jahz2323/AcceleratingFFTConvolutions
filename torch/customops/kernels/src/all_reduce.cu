#include "all_reduce.cuh"


__global__ void gpu_all_reduce(int* sum, int* data, int n){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    int local_temp = 0;

    for (int i = idx; i < n; i += stride) {
        local_temp += data[i];
    }
    atomicAdd(sum, local_temp);
}

void cpu_all_reduce(int* sum, int* data, int n){
    int temp_sum = 0;
    for (int i = 0; i < n; ++i) {
        temp_sum += data[i];    
    }
    *sum = temp_sum;
}


static const int BLOCKX_DIM = 256;
torch::Tensor all_reduce_launcher(torch::Tensor input){
    torch::Device device(torch::kCUDA,0);
    torch::Tensor output = torch::zeros({1}, torch::device(torch::kCUDA).dtype(torch::kInt));
    if (input.device() == device){
        std::cout<<"GPU all reduce launched!"<<std::endl;
        output = output.to(device); 
        dim3 blockSize(BLOCKX_DIM);
        dim3 gridSize((input.size(0) + BLOCKX_DIM - 1) / BLOCKX_DIM);
        const cudaStream_t stream = at::cuda::getCurrentCUDAStream(); 
        gpu_all_reduce<<<gridSize, blockSize, 0, stream>>>( output.data_ptr<int>(),input.data_ptr<int>(), input.size(0));
    }
    else{
        std::cout<<"CPU all reduce launched!"<<std::endl;
        cpu_all_reduce(output.data_ptr<int>(), input.data_ptr<int>(), input.size(0));
    }
    
    return output;
}