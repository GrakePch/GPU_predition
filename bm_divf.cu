#include <stdio.h>
#include <cuda.h>

# define N 1e7

__global__ void microbenchmark(float *input, float *output) {
    float temp = *input;

    for (int i = 0; i < N; i++) {
        temp /= 1.234f; // Example computation
    }
    *output = temp;
}

int main(int argc, char *argv[]) {
    
    if (argc != 3) {
        printf("usage: <freq in GHz> <GPU id>\n");
        exit(1);
    }

    
    float f = atof(argv[1]);
    printf("GPU frequency %f Ghz.\n", f);

    float *d_input, *d_output;
    float h_input = 1.0f, h_output;

    cudaSetDevice(atoi(argv[2]));

    cudaMalloc(&d_input, sizeof(float));
    cudaMalloc(&d_output, sizeof(float));

    cudaMemcpy(d_input, &h_input, sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);
    microbenchmark<<<1, 1>>>(d_input, d_output);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    cudaMemcpy(&h_output, d_output, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_input);
    cudaFree(d_output);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    printf("Time elapsed: %f ms\n", milliseconds);
    float cycles = milliseconds / 1000 / N * f * 1e9;
    printf("Cycle: %f\n", cycles);
    return 0;
}
