
float* vectorGPUAdd (float* a, float* b, int n, int p);


__global__ void vectorAdd (float *out, float *a, float *b, int n, int p);

