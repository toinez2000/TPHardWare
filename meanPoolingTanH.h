__device__ float activation_tanh(float M);


__global__ void meanPooling(float* input, float* output, int SxI, int SzI);


float* vectorGPUMeanPooling(float*input,int SxI,int SzI);
