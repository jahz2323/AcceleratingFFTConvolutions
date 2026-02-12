
#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <cuda_runtime.h>

__global__ void VecAdd(float* A, float* B, float* C) {
    int i = threadIdx.x;
    printf("Thread %d: Adding A[%d] and B[%d]\n", i, i, i); 
    C[i] = A[i] + B[i];
}
void add(int N, float *x, float *y, float *z){
    for (int i = 0; i < N; i++) {
        z[i] = x[i] + y[i];
    }
}
int main(){
    int N = 1<<30;
    
    float*x = new float[N]; 
    float*y = new float[N];
    float*z = new float[N];
    for (int i = 0; i < N; i++) {
        x[i] = static_cast<float>(i);
        y[i] = static_cast<float>(i * 2);
    }
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    add(N, x, y, z);
    cudaEventRecord(stop);
    // map device mem back to host
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Time taken: " << milliseconds << " ms" << std::endl;

    // testing with GPU kernel
    z = new float[N]; // reset z
    cudaEventRecord(start);
    VecAdd<<<1, N>>>(x, y, z);
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    std::cout << "Time taken (GPU): " << milliseconds << " ms" << std::endl;

    delete[] x;
    delete[] y;
    delete[] z;
    return 0;
}