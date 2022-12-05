#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>

#include "vector_multi_N_P.h"

#define M 2000
#define P 2000
#define N M*P
#define MAX_ERR 1e-6

__global__ void MatriceNxN_multi(float *out, float *a, float *b, int n) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
	int Size = n*n;

// Handling arbitrary vector size
 	if (tid < Size){
		for(int i =0; i < n; i++){
			out[tid] = out[tid]+ a[blockIdx.x*blockDim.x+i] * b[i*blockDim.x+threadIdx.x];

		}
    	}
}


__host__ float* multimatriceGPU(float* a,float* b, int n){

	float *out;
	float *d_a, *d_b, *d_out;

	out = (float*)malloc(sizeof(float) * n*n);

	cudaMalloc((void**)&d_a, sizeof(float)*n*n);
    	cudaMalloc((void**)&d_b, sizeof(float)*n*n);
    	cudaMalloc((void**)&d_out, sizeof(float)*n*n);

	
    	cudaMemcpy(d_a, a, sizeof(float) * n*n, cudaMemcpyHostToDevice);
    	cudaMemcpy(d_b, b, sizeof(float) * n*n, cudaMemcpyHostToDevice);

	// Main function
    	//int block_size = atoi(argv[2]);
    	//int grid_size = atoi(argv[1]);
		int block_size = n;
    	int grid_size = n;
    	MatriceNxN_multi<<<grid_size,block_size>>>(d_out, d_a, d_b, n);

    	cudaMemcpy(out, d_out, sizeof(float)*n*n, cudaMemcpyDeviceToHost);
	
	
    	cudaFree(d_a);
    	cudaFree(d_b);
    	cudaFree(d_out);

	return out;

}




/*
int main1(int argc, char *argv[]){

    float *a, *b;


    // Allocate memory
    a   = (float*)malloc(sizeof(float) * N);
    b   = (float*)malloc(sizeof(float) * N);


    // Initialize array
    for(int i = 0; i < N; i++){
        a[i] = 1.0f;
        b[i] = 2.0f;
    }


float* out =  multimatriceGPU(a,b, N);


    // Verification
    for(int i = 0; i < N; i++){
        assert(fabs(out[i] - a[i] - b[i]) < MAX_ERR);
    }

   
    printf("out[0] = %f\n", out[0]);
    printf("PASSED\n");


    free(a);
    free(b);
    free(out);

	return 0;
}
*/