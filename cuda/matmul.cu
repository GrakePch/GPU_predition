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

void matrixMulHost(float* h_A, float* h_B, float* h_C, int n) {
    size_t bytes = n * n * sizeof(float);

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((n + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (n + threadsPerBlock.y - 1) / threadsPerBlock.y);

    matrixMulKernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, n);

    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main(int argc, char * argv[]) {
    int n;
    
    if(argc != 2)
    {
        fprintf(stderr, "usage: matmul dimension\n");
        exit(1);
    }

    n = (unsigned int) atoi(argv[1]);

    size_t bytes = n * n * sizeof(float);

    double time_taken;
    clock_t start, end;   // to meaure the time taken by a specific part of code

    float* h_A = (float*)malloc(bytes);
    float* h_B = (float*)malloc(bytes);
    float* h_C = (float*)malloc(bytes);

    for (int i = 0; i < n * n; i++) {
        h_A[i] = (float)rand() / RAND_MAX;
        h_B[i] = (float)rand() / RAND_MAX;
    }

    start = clock();
    matrixMulHost(h_A, h_B, h_C, n);
    end = clock();


    
    time_taken = ((double)(end - start))/ CLOCKS_PER_SEC;
    
    printf("Time taken = %lf\n", time_taken);

    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
