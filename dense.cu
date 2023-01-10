#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "functionMatrix.h"
#include "dense.h"




__global__ void Dense(float *input,float *weight,float *output,int SxI,int Sxo){
    
  int tabIP = blockIdx.x ;
  int ip = threadIdx.x;

  output[tabIP] += input[ip]*weight[tabIP*SxI+ip];

}


__global__ void DTanH(float *input,int SxI){
    
  
  int ip = threadIdx.x;

  input[ip] = tanh(input[ip]);

}

__global__ void softMax(float *input,int SxI,float *sum)
{
  int ip = threadIdx.x;

  input[ip] = input[ip]/sum[0];
}


__global__ void sumExpo(float *input,int SxI,float* sum)
{
  int ip = threadIdx.x;
  input[ip] = exp(input[ip]);
  sum[0] += input[ip];
}







//-----------------------------------------------------------------------------------

float* vectorGPUDense (float* input, float* Weight,float *output,int SxI,int Sxo,int ActiveFunction)
{

  float *sum;
  sum = 0;
	float *d_input, *d_Weight, *d_out, *d_sum;




	cudaMalloc((void**)&d_input, sizeof(float)*SxI);
    	cudaMalloc((void**)&d_Weight, sizeof(float)*SxI);
    	cudaMalloc((void**)&d_out, sizeof(float)*Sxo);
      cudaMalloc((void**)&d_sum, sizeof(float));
	
    	cudaMemcpy(d_Weight, Weight, sizeof(float) *SxI, cudaMemcpyHostToDevice);
    	cudaMemcpy(d_input, input, sizeof(float) *SxI, cudaMemcpyHostToDevice);
      cudaMemcpy(d_out,output,sizeof(float) *Sxo,cudaMemcpyHostToDevice);
      cudaMemcpy(d_sum,sum,sizeof(float),cudaMemcpyHostToDevice);

	// Main function

    dim3 blocks(Sxo); 
    dim3 threadsPerBlock(SxI);
 
      

    	 Dense<<<threadsPerBlock,blocks>>>(d_input, d_Weight, d_out, SxI,  Sxo);   //SIZE_C1_kernel
 
      if(ActiveFunction==0){DTanH<<<1,Sxo>>>(d_out,Sxo);} //TanH
      else{
          
          sumExpo<<<1,Sxo>>>(d_out,Sxo,d_sum);
          softMax<<<1,Sxo>>>(d_out,Sxo,d_sum);
      }

    	cudaMemcpy(output, d_out, sizeof(float)*Sxo, cudaMemcpyDeviceToHost);
	
      cudaFree(d_sum);
    	cudaFree(d_Weight);
    	cudaFree(d_input);
    	cudaFree(d_out);


}
