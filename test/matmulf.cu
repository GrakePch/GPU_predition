#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

__global__ void matrixMulKernel(float* A, float* B, float* C, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;  
    int col = blockIdx.x * blockDim.x + threadIdx.x; 

    if (row < n && col < n) {
        float value = 0.0f;
        for (int k = 0; k < n; k++) {
            value += A[row * n + k] * B[k * n + col];
        }
        C[row * n + col] = value;
    }
}



int main(int argc, char * argv[]) {
    int n;
    
    if(argc != 3)
    {
        fprintf(stderr, "need arguments: matDim gpuId\n");
        exit(1);
    }

    n = (unsigned int) atoi(argv[1]);
    cudaSetDevice(atoi(argv[2]));

    size_t bytes = n * n * sizeof(float);

    float* h_A = (float*)malloc(bytes);
    float* h_B = (float*)malloc(bytes);
    float* h_C = (float*)malloc(bytes);

    for (int i = 0; i < n * n; i++) {
        h_A[i] = (float)rand() / RAND_MAX;
        h_B[i] = (float)rand() / RAND_MAX;
    }

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int it = 20;
    float milliseconds = 0;

    
    for (int i = 0; i < it; ++i) {

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (n + threadsPerBlock.y - 1) / threadsPerBlock.y);

    cudaEventRecord(start);
    matrixMulKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, n);
    cudaEventRecord(stop);

    
    cudaEventSynchronize(stop);

    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    printf("%f\n", ms/1e3);
    if (i > 0)  // use first run as warm-up to increase accuracy
        milliseconds += ms;
    }
    
    milliseconds /= it - 1;

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    
    printf("GPU time = %lf secs\n", milliseconds / 1e3);

    
    cudaDeviceSynchronize();  // block host till device is done.
    
    free(h_A);
    free(h_B);
    free(h_C);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);




    return 0;
}
