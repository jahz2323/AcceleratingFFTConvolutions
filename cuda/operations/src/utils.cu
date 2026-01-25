#include "utils.cuh"



/**
    @author https://github.com/dbids-EC527/fft/blob/master/base_code/fft_2d.c
*/
__global__ void utils::bitreversal(int n, void* storage){
    int i, j;
    for (i=1, j=0; i < n; i++){
        int bit = n >>1; 
        for(; j & bit; bit>>=1){
            j ^= bit;
        }
        j ^= bit;
        cuComplex temp;
        if(i < j){
            // swap
            temp = static_cast<cuComplex*>(storage)[i];
            static_cast<cuComplex*>(storage)[i] = static_cast<cuComplex*>(storage)[j];
            static_cast<cuComplex*>(storage)[j] = temp;
        }
    } 
}

__global__ void utils::bitreversal(int width, int height, cuComplex* data){
    auto idx  = blockIdx.x * blockDim.x + threadIdx.x;
    auto idy  = blockIdx.y * blockDim.y + threadIdx.y;

    if (idx < width && idy < height){
        unsigned int rev_n = 0; 
        int temp_x = idx;
        int bits = static_cast<int>(log2f((float)width));
        for(int i = 0; i < bits; i++){
            rev_n = (rev_n << 1) | (temp_x & 1);
            temp_x >>= 1;
        }
        if  (rev_n > idx){
            // swap
            cuComplex temp = data[idy * width + idx];
            data[idy * width + idx] = data[idy * width + rev_n];
            data[idy * width + rev_n] = temp;
        }
    }
}

__global__ void utils::float2complex(int width, int height, float* input, cuComplex* output){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if(idx >= width || idy >= height) return;

    int index = idy * width + idx;
    float real_value = static_cast<float*>(input)[index];
    (output)[index] = make_cuComplex(real_value, 0.0f);
}

__global__ void utils::complex2float(int width, int height, cuComplex* input, float* output){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if(idx >= width || idy >= height) return;

    int index = idy * width + idx;
    float scale = 1.0f / (width * height);
    cuComplex cvalue = (input)[index];
    static_cast<float*>(output)[index] = cuCrealf(cvalue) * scale;
}

__global__ void utils::copy(float* odata, float* idata){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int width = gridDim.x * blockDim.x;
    int index = idy * width + idx;
    odata[index] = idata[index];
}

__global__ void utils::copyComplex(cuComplex* odata, cuComplex* idata){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    int width = gridDim.x * blockDim.x;
    int index = idy * width + idx;
    odata[index] = idata[index];
}

int utils::nextPowerOfTwo(int n){
    int count = 0;
    // First n in the below condition is for the case where n is 0
    if (n && !(n & (n - 1)))
        return n;

    while( n != 0){
        n >>= 1;
        count += 1;
    }
    return 1 << count;
}

__global__ void utils::naivetranspose(int width, int height, cuComplex* input, cuComplex* output){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if(idx >= width || idy >= height) return;

    int in_index = idy * width + idx;
    int out_index = idx * height + idy;
    output[out_index] = input[in_index];
}