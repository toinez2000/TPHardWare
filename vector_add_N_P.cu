#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "vector_add_N_P.h"
#define M 2000
#define P 2000
#define N M*P
#define MAX_ERR 1e-6

__global__ void vectorAdd (float *out, float *a, float *b, int n, int p) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int Size = n*p;
    // Handling arbitrary vector size
    if (tid < Size){
        out[tid] = a[tid] + b[tid];
    }
}



float* vectorGPUAdd (float* a, float* b, int n, int p)
{

	float *out;
	float *d_a, *d_b, *d_out;

	out = (float*)malloc(sizeof(float) * n*p);

	cudaMalloc((void**)&d_a, sizeof(float)*n*p);
    	cudaMalloc((void**)&d_b, sizeof(float)*n*p);
    	cudaMalloc((void**)&d_out, sizeof(float)*n*p);

	
    	cudaMemcpy(d_a, a, sizeof(float) * n*p, cudaMemcpyHostToDevice);
    	cudaMemcpy(d_b, b, sizeof(float) * n*p, cudaMemcpyHostToDevice);

	// Main function
    	//int block_size = atoi(argv[2]);
    	//int grid_size = atoi(argv[1]);
	int block_size = p;
    	int grid_size = n;
    	vectorAdd<<<grid_size,block_size>>>(d_out, d_a, d_b, n,p);

    	cudaMemcpy(out, d_out, sizeof(float)*n*p, cudaMemcpyDeviceToHost);
	
	
    	cudaFree(d_a);
    	cudaFree(d_b);
    	cudaFree(d_out);

	return out;

}

/*
int main(int argc, char *argv[]){

    float *a, *b, *out;
 

    // Allocate memory
    a   = (float*)malloc(sizeof(float) * N);
    b   = (float*)malloc(sizeof(float) * N);
    

    // Initialize array
    for(int i = 0; i < N; i++){
        a[i] = 1.0f;
        b[i] = 2.0f;
    }


	//function add
	out = vectorGPUAdd(a, b, M, P);

    // Verification
    for(int i = 0; i < N; i++){
        assert(fabs(out[i] - a[i] - b[i]) < MAX_ERR);
    }

    //printf("Grid size : %d\n", grid_size);
    //printf("Block size : %d\n", block_size);
    printf("out[0] = %f\n", out[0]);
    printf("PASSED\n"); 


    free(a);
    free(b);
    free(out);
}
*/