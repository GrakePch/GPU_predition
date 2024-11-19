#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void kernel(float *, float *, float, int);

int main(int argc, char *argv[]) {
    int i;
    int num = 0;        // number of elements in the arrays
    float *a, *c;     // arrays at host
    float *ad, *cd;  // arrays at device

    if (argc != 3) {
        printf("need arguments: numelements gpuId\n");
        exit(1);
    }

    num = atoi(argv[1]);
    cudaSetDevice(atoi(argv[2]));

    a = (float *)malloc(num * sizeof(float));
    if (!a) {
        printf("Cannot allocate array a with %d elements\n", num);
        exit(1);
    }

    c = (float *)malloc(num * sizeof(float));
    if (!c) {
        printf("Cannot allocate array c with %d elements\n", num);
        exit(1);
    }

    // Fill out arrays a and b with some random numbers
    srand(time(0));
    for (i = 0; i < num; i++) {
        a[i] = rand() % num;
    }

    // Now zero C[] in preparation for GPU version
    for (i = 0; i < num; i++)
        c[i] = 0;

    int numblocks;
    int threadsperblock = 256;

    if ((num % threadsperblock) == 0)
        numblocks = num / threadsperblock;
    else
        numblocks = (num / threadsperblock) > 0 ? (num / threadsperblock) + 1 : 1;

    printf("GPU: %d blocks of %d threads each\n", numblocks, threadsperblock);

    // assume a block can have THREADS threads
    dim3 grid(numblocks, 1, 1);
    dim3 block(threadsperblock, 1, 1);

    cudaMalloc((void **)&ad, num * sizeof(float));
    if (!ad) {
        printf("cannot allocated array ad of %d elements\n", num);
        exit(1);
    }

    cudaMalloc((void **)&cd, num * sizeof(float));
    if (!cd) {
        printf("cannot allocated array cd of %d elements\n", num);
        exit(1);
    }

    // mov a and b to the device
    cudaMemcpy(ad, a, num * sizeof(float), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Launch the kernel
    cudaEventRecord(start);
    kernel<<<numblocks, threadsperblock>>>(ad, cd, 1.234, num);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // bring data back
    cudaMemcpy(c, cd, num * sizeof(float), cudaMemcpyDeviceToHost);

    printf("GPU time = %lf secs\n", milliseconds / 1e3);

    cudaDeviceSynchronize();  // block host till device is done.

    free(a);
    free(c);

    cudaFree(ad);
    cudaFree(cd);
}

__global__ void kernel(float *a, float *c, float d, int n) {
    int index;

    index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < n)
        c[index] = a[index] * d;
}
