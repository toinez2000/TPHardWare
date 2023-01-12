
/**
* function activation_tanh appli tanH to output layer
**/
__device__ float activation_tanh(float M);


/**
* function meanPooling with GPU
*return output matrix  size SxI*SxI*SzI/4
**/


__global__ void meanPooling(float* input, float* output, int SxI, int SzI);


/**
* function vectorGPUMeanPooling manage Layer meanPooling
**/

float* vectorGPUMeanPooling(float*input,int SxI,int SzI);
