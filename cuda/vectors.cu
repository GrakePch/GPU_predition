#include <cuda.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define RANGE 17.78

/*** TODO: insert the declaration of the kernel function below this line ***/
__global__ void multVector(float *a, float *b, float *c, int n, int opPerTh);

/**** end of the kernel declaration ***/

int main(int argc, char *argv[]) {
    int n = 0;            // number of elements in the arrays
    int i;                // loop index
    float *a, *b, *c;     // The arrays that will be processed in the host.
    float *temp;          // array in host used in the sequential code.
    float *ad, *bd, *cd;  // The arrays that will be processed in the device.
    clock_t start, end;   // to meaure the time taken by a specific part of code

    if (argc != 2) {
        printf("usage:  ./vectorprog n\n");
        printf("n = number of elements in each vector\n");
        exit(1);
    }

    n = atoi(argv[1]);
    printf("Each vector will have %d elements\n", n);

    // Allocating the arrays in the host

    if (!(a = (float *)malloc(n * sizeof(float)))) {
        printf("Error allocating array a\n");
        exit(1);
    }

    if (!(b = (float *)malloc(n * sizeof(float)))) {
        printf("Error allocating array b\n");
        exit(1);
    }

    if (!(c = (float *)malloc(n * sizeof(float)))) {
        printf("Error allocating array c\n");
        exit(1);
    }

    if (!(temp = (float *)malloc(n * sizeof(float)))) {
        printf("Error allocating array temp\n");
        exit(1);
    }

    // Fill out the arrays with random numbers between 0 and RANGE;
    srand((unsigned int)time(NULL));
    for (i = 0; i < n; i++) {
        a[i] = ((float)rand() / (float)(RAND_MAX)) * RANGE;
        b[i] = ((float)rand() / (float)(RAND_MAX)) * RANGE;
        c[i] = ((float)rand() / (float)(RAND_MAX)) * RANGE;
        temp[i] = c[i];  // temp is just another copy of C
    }

    // The sequential part
    start = clock();
    for (i = 0; i < n; i++)
        temp[i] += a[i] * b[i];
    end = clock();
    printf("Total time taken by the sequential part = %lf\n", (double)(end - start) / CLOCKS_PER_SEC);

    /******************  The start GPU part: Do not modify anything in main() above this line  ************/
    // The GPU part

    /* TODO: in this part you need to do the following:
            1. allocate ad, bd, and cd in the device
            2. send a, b, and c to the device
            */
    int numBlocks = 4;
    int threadsPerBlock = 500;
    int opPerThread;
    if ((n % (numBlocks * threadsPerBlock)) == 0)
        opPerThread = n / (numBlocks * threadsPerBlock);
    else
        opPerThread = n / (numBlocks * threadsPerBlock) + 1;

    start = clock();

    cudaMalloc((void **)&ad, n * sizeof(float));
    if (!ad) {
        printf("Error allocating array ad\n");
        exit(1);
    }
    cudaMalloc((void **)&bd, n * sizeof(float));
    if (!bd) {
        printf("Error allocating array bd\n");
        exit(1);
    }
    cudaMalloc((void **)&cd, n * sizeof(float));
    if (!cd) {
        printf("Error allocating array cd\n");
        exit(1);
    }

    cudaMemcpy(ad, a, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(bd, b, n * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(cd, c, n * sizeof(float), cudaMemcpyHostToDevice);


    /* TODO:
            3. write the kernel, call it: vecGPU
            4. call the kernel (the kernel itself will be written at the comment at the end of this file),
               you need to decide about the number of threads, blocks, etc and their geometry.
            */
    multVector<<<numBlocks, threadsPerBlock>>>(ad, bd, cd, n, opPerThread);
    /* TODO:
            5. bring the cd array back from the device and store it in c array (declared earlier in main)
            6. free ad, bd, and cd
    */
    cudaMemcpy(c, cd, n * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(ad);
    cudaFree(bd);
    cudaFree(cd);
    
    end = clock();

    printf("Total time taken by the GPU part = %lf\n", (double)(end - start) / CLOCKS_PER_SEC);
    /******************  The end of the GPU part: Do not modify anything in main() below this line  ************/

    // checking the correctness of the GPU part
    for (i = 0; i < n; i++)
        if (fabs(temp[i] - c[i]) >= 0.009)  // compare up to the second degit in floating point
            printf("Element %d in the result array does not match the sequential version\n", i);

    // Free the arrays in the host
    free(a);
    free(b);
    free(c);
    free(temp);

    return 0;
}

/**** TODO: Write the kernel itself below this line *****/
__global__ void multVector(float *a, float *b, float *c, int n, int opPerTh) {
    int index;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    for (int op = 0; op < opPerTh; ++op) {
        index = tid * opPerTh + op;
        if (index < n)
            c[index] += a[index] * b[index];
    }
}
