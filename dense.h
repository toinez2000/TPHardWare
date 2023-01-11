<<<<<<< HEAD
float* vectorGPUDense (float* input, float* Weight,float *output,int SxI,int Sxo,int ActiveFunction);
=======
float* vectorGPUDense (float* input, float* Weight,int SxI,int Sxo,int ActiveFunction);
>>>>>>> dvt


//-----------------------------------------------------------------------------------------------

__global__ void Dense(float *input,float *weight,float *output,int SxI,int Sxo);

__global__ void DTanH(float *input,int SxI);




//-------------
__global__ void softMax(float *input,int SxI,float *sum);


__global__ void sumExpo(float *input,int SxI,float* sum);
