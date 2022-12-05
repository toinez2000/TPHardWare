int main2();
float* vectorGPUConv1 (float* Kernel, float* input);
__global__ void convolutionLayer(float* C1_data,float* C1_kernel, float* raw_data);