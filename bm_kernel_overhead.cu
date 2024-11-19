#include <cuda.h>
#include <stdio.h>

__global__ void emptyKernel() {
}

int main() {
    cudaEvent_t start, stop;
    float elapsedTime;
    
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    int threadsPerBlock = 256; 
    int blocks;

    FILE *outputFile = fopen("kernel_launch_overhead.csv", "w");
    if (!outputFile) {
        fprintf(stderr, "Failed to open output file.\n");
        return -1;
    }
    fprintf(outputFile, "NumThreads,Overhead(s)\n");

    printf("Measuring kernel launch overhead...\n");
    
    // Warm-up
    emptyKernel<<<1, threadsPerBlock>>>();
    cudaDeviceSynchronize();

    int iterations = 10;

    for (long long numThreads = 1e6; numThreads <= 8e6; numThreads += 1e6) {
        
        blocks = (numThreads + threadsPerBlock - 1) / threadsPerBlock;

        float avgOverhead = 0;
        for (int i = 0; i < iterations ;++i)
        {
            cudaEventRecord(start);
            emptyKernel<<<blocks, threadsPerBlock>>>();
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            cudaEventElapsedTime(&elapsedTime, start, stop); // milliseconds

            avgOverhead += elapsedTime / 1000.0;
        }

        avgOverhead /= iterations;

        printf("Threads: %lld, Overhead: %.8f seconds\n", numThreads, avgOverhead);
        fprintf(outputFile, "%lld,%.8f\n", numThreads, avgOverhead);
    }

    for (long long numThreads = 1000; numThreads <= 1e7; numThreads *= 10) {
        
        blocks = (numThreads + threadsPerBlock - 1) / threadsPerBlock;

        float avgOverhead = 0;
        for (int i = 0; i < iterations ;++i)
        {
            cudaEventRecord(start);
            emptyKernel<<<blocks, threadsPerBlock>>>();
            cudaEventRecord(stop);
            cudaEventSynchronize(stop);

            cudaEventElapsedTime(&elapsedTime, start, stop); // milliseconds

            avgOverhead += elapsedTime / 1000.0;
        }

        avgOverhead /= iterations;

        printf("Threads: %lld, Overhead: %.8f seconds\n", numThreads, avgOverhead);
        fprintf(outputFile, "%lld,%.8f\n", numThreads, avgOverhead);
    }

    fclose(outputFile);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("Results saved to 'kernel_launch_overhead.csv'.\n");
    return 0;
}