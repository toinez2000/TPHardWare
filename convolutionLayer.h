

__global__ void conv3D(float* input, float* kernel, float* output, int SxI, int SzI, int SxK, int SzK);

float* vectorGPUConv1 (float* Kernel, float* input,int SxI,int SzI,int SxK,int SzK);
