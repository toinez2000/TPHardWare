#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "functionMatrix.h"
#include "dense.h"

//fonction add bias 
__global__ void addBias(float *input,float *bias){
    
  int ip = threadIdx.x;

<<<<<<< HEAD
  output[ip] += bias[ip];
=======
  input[ip] += bias[ip];
>>>>>>> dvt

}



__global__ void Dense(float *input,float *weight,float *output,int SxI,int Sxo){
    
  int tabIP = blockIdx.x ; //for each output you add multiplication of weight and input

  for(int ip =0;ip < SxI;ip++){
    output[tabIP] += input[ip]*weight[tabIP*SxI+ip];}
>>>>>>> dvt

}

// active function TanH
__global__ void DTanH(float *input,int SxI){
    
  
  int ip = threadIdx.x;

  input[ip] = tanh(input[ip]);

}

//active function softMax
__global__ void softMax(float *input,int SxI,float sum)
{
  int ip = threadIdx.x;

  input[ip] = input[ip]/sum;
}

//function calcul the exp of each output
__global__ void Expo(float *input,int SxI)
{
  int ip = threadIdx.x;
  input[ip] = exp(input[ip]);
>>>>>>> dvt
}







//-----------------------------------------------------------------------------------

<<<<<<< HEAD
float* vectorGPUDense (float* input, float* Weight,float *output,int SxI,int Sxo,int ActiveFunction)
{

  float *sum;
  sum = 0;
	float *d_input, *d_Weight, *d_out, *d_sum;




	cudaMalloc((void**)&d_input, sizeof(float)*SxI);
    	cudaMalloc((void**)&d_Weight, sizeof(float)*SxI*Sxo+Sxo);
    	cudaMalloc((void**)&d_out, sizeof(float)*Sxo);
      cudaMalloc((void**)&d_sum, sizeof(float));
	
    	cudaMemcpy(d_Weight, Weight, sizeof(float) *SxI*Sxo+Sxo, cudaMemcpyHostToDevice);
    	cudaMemcpy(d_input, input, sizeof(float) *SxI, cudaMemcpyHostToDevice);
      cudaMemcpy(d_out,output,sizeof(float) *Sxo,cudaMemcpyHostToDevice);
      cudaMemcpy(d_sum,sum,sizeof(float),cudaMemcpyHostToDevice);
=======
float* vectorGPUDense (float* input, float* Weight,int SxI,int Sxo,int ActiveFunction)
{

  float sum, *output, *output0;

output= (float*)malloc(sizeof(float) *Sxo);
output0= (float*)malloc(sizeof(float) *Sxo);

  sum = 0;
	float *d_input, *d_Weight, *d_out;

    

//malloc GPU
	cudaMalloc((void**)&d_input, sizeof(float)*SxI);
    	cudaMalloc((void**)&d_Weight, sizeof(float)*(SxI*Sxo+Sxo));
    	cudaMalloc((void**)&d_out, sizeof(float)*Sxo);
	
    	cudaMemcpy(d_Weight, Weight, sizeof(float) *(SxI*Sxo+Sxo), cudaMemcpyHostToDevice);
    	cudaMemcpy(d_input, input, sizeof(float) *SxI, cudaMemcpyHostToDevice);
      	cudaMemcpy(d_out,output,sizeof(float) *Sxo,cudaMemcpyHostToDevice);

	// Main function

    dim3 blocks(Sxo); 
<<<<<<< HEAD
    dim3 threadsPerBlock(SxI);
 
      

    	 Dense<<<threadsPerBlock,blocks>>>(d_input, d_Weight, d_out, SxI,  Sxo);   //SIZE_C1_kernel
 	 addBias<<<Sxo,1>>>(d_out,d_Weight+SxI*Sxo);
      if(ActiveFunction==0){DTanH<<<1,Sxo>>>(d_out,Sxo);} //TanH
      else{
          
          sumExpo<<<1,Sxo>>>(d_out,Sxo,d_sum);
          softMax<<<1,Sxo>>>(d_out,Sxo,d_sum);
=======
 
      

    	 Dense<<<blocks,1>>>(d_input, d_Weight, d_out, SxI,  Sxo);   //calcul each output of dense without bias
       cudaDeviceSynchronize();
 	    addBias<<<Sxo,1>>>(d_out,d_Weight+SxI*Sxo); //add bias
      cudaDeviceSynchronize();
      if(ActiveFunction==0){
      DTanH<<<1,Sxo>>>(d_out,Sxo); 
      cudaDeviceSynchronize();
      
      } //TanH
      else{
          
          Expo<<<1,Sxo>>>(d_out,Sxo);
          cudaDeviceSynchronize();
          cudaMemcpy(output0, d_out, sizeof(float)*Sxo, cudaMemcpyDeviceToHost);
          for(int j=0;j<Sxo;j++){
            sum =  sum+output0[j];
          }
          

          softMax<<<1,Sxo>>>(d_out,Sxo,sum);
          cudaDeviceSynchronize();
>>>>>>> dvt
      }

    	cudaMemcpy(output, d_out, sizeof(float)*Sxo, cudaMemcpyDeviceToHost);
	//free
    	cudaFree(d_Weight);
    	cudaFree(d_input);
    	cudaFree(d_out);

<<<<<<< HEAD

=======
  return output;
  
>>>>>>> dvt
}
