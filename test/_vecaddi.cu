#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void kernel(int *, int *, int *, int);

int main(int argc, char *argv[]) {
    int i;
    int num = 0;        // number of elements in the arrays
    int *a, *b, *c;     // arrays at host
    int *ad, *bd, *cd;  // arrays at device

    if (argc != 3) {
        printf("need arguments: numelements gpuId\n");
        exit(1);
    }

    num = atoi(argv[1]);
    cudaSetDevice(atoi(argv[2]));

    a = (int *)malloc(num * sizeof(int));
    if (!a) {
        printf("Cannot allocate array a with %d elements\n", num);
        exit(1);
    }

    b = (int *)malloc(num * sizeof(int));
    if (!b) {
        printf("Cannot allocate array b with %d elements\n", num);
        exit(1);
    }

    c = (int *)malloc(num * sizeof(int));
    if (!c) {
        printf("Cannot allocate array c with %d elements\n", num);
        exit(1);
    }

    // Fill out arrays a and b with some random numbers
    srand(time(0));
    for (i = 0; i < num; i++) {
        a[i] = rand() % num;
        b[i] = rand() % num;
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

    cudaMalloc((void **)&ad, num * sizeof(int));
    if (!ad) {
        printf("cannot allocated array ad of %d elements\n", num);
        exit(1);
    }

    cudaMalloc((void **)&bd, num * sizeof(int));
    if (!bd) {
        printf("cannot allocated array bd of %d elements\n", num);
        exit(1);
    }

    cudaMalloc((void **)&cd, num * sizeof(int));
    if (!cd) {
        printf("cannot allocated array cd of %d elements\n", num);
        exit(1);
    }

    // mov a and b to the device
    cudaMemcpy(ad, a, num * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(bd, b, num * sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Launch the kernel
    cudaEventRecord(start);
    kernel<<<numblocks, threadsperblock>>>(ad, bd, cd, num);
    cudaEventRecord(stop);

    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // bring data back
    cudaMemcpy(c, cd, num * sizeof(int), cudaMemcpyDeviceToHost);

    printf("GPU time = %lf secs\n", milliseconds / 1e3);

    cudaDeviceSynchronize();  // block host till device is done.

    free(a);
    free(b);
    free(c);

    cudaFree(ad);
    cudaFree(bd);
    cudaFree(cd);
}

__global__ void kernel(int *a, int *b, int *c, int n) {
    int index;

    index = (blockIdx.x * blockDim.x) + threadIdx.x;

    if (index < n)
        c[index] = a[index] + b[index];
}
