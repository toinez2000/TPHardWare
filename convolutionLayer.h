/**
* function conv3D
* param tab input layer size SxI*SxI*SzI
* param tab kernel Weight de la convolution size SxK*SxK*SzK
* param SxI size x input
* param SzI size deep input
* param SxK size x kernel
* param SzK size deep kernel
* param output tab output Layer size (SxI-SxK+1)*(SxI-SxK+1)*SxK*SxI
**/

__global__ void conv3D(float* input, float* kernel, float* output, int SxI, int SzI, int SxK, int SzK);

/**
* function vectorGPUConv1 preparation GPU and manage Layer conv
* param tab input layer size SxI*SxI*SzI
* param tab kernel Weight+bias de la convolution size SxK*SxK*SzK+SzK
* param SxI size x input
* param SzI size deep input
* param SxK size x kernel
* param SzK size deep kernel
* return output tab output Layer size (SxI-SxK+1)*(SxI-SxK+1)*SxK*
**/


float* vectorGPUConv1 (float* Kernel, float* input,int SxI,int SzI,int SxK,int SzK);
