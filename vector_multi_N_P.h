__host__ float* multimatriceGPU(float* a,float* b, int n);
__global__ void MatriceNxN_multi(float *out, float *a, float *b, int n);