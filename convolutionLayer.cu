#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "functionMatrix.h"
#include "convolutionLayer.h"

__global__ void conv3D(float* input, float* kernel, float* output, int SxI, int SzI, int SxK, int SzK) {
  // Calculer les indices de l'élément de sortie courant
  int xo = blockIdx.x * blockDim.x; // ligne de l'image de sortie pour une profondeur donné 
  int yo = threadIdx.x; // element de la ligne
  int Sxo = (SxI-SxK+1)* (SxI-SxK+1);

  int zI = blockIdx.z ;   //profondeur du kernel à utiliser 
  int zK = threadIdx.z; // proffondeur de l'image input à utilisé 

  // Initialiser la valeur de l'élément de sortie à 0
  float value = 0.0;



  
		// Appliquer le filtre sur chaque élément de l'image
	  for (int i = 0; i < SxK; i++) {
	    for (int j = 0; j < SxK; j++) {
	        // Calculer les indices de l'élément de l'image à utiliser
			

			int decalage = blockIdx.x*(SxK-1);


          int xI = xo+ i*SxI + decalage +zI*(SxI*SxI);  // lien entre xI et x0
	        int yI = yo + j;
          	// Appliquer le filtre à l'élément de l'image courant
           value += input[xI+ yI] * kernel[i*SxK + j + zK*SxK*SxK];
          //printf("xo = %d,yo = %d, zI = %d, zk =%d , indiceInput = %d indiceKernel= %d \n",xo,yo,zI,zK,xI+ yI,i*SxK + j + zK*SxK*SxK);
          //printf("xo = %f ,indiceK = %d \n",kernel[i*SxK + j + zK*SxK*SxK],i*SxK + j + zK*SxK*SxK);
	      }
	    }

  	  // Enregistrer la valeur de l'élément de sortie
    output[xo+ yo + (zK) *(SxI-SxK+1)*(SxI-SxK+1)] += value;
  
         
}



float* vectorGPUConv1 (float* Kernel, float* input,int SxI,int SzI,int SxK,int SzK)
{

	float *out;
	float *d_input, *d_Kernel, *d_out;
  int Sxo = SxI-SxK+1;
  int Szo = SzK*SzI;

	out = (float*)malloc(sizeof(float) * Sxo*Sxo*Szo);

	cudaMalloc((void**)&d_input, sizeof(float)*SxI*SxI*SzI);
    	cudaMalloc((void**)&d_Kernel, sizeof(float)*SxK*SxK*SzK);
    	cudaMalloc((void**)&d_out, sizeof(float)*Sxo*Sxo*Szo);

	
    	cudaMemcpy(d_Kernel, Kernel, sizeof(float) *SxK*SxK*SzK, cudaMemcpyHostToDevice);
    	cudaMemcpy(d_input, input, sizeof(float) *SxI*SxI*SzI, cudaMemcpyHostToDevice);

	// Main function
    	//int block_size = atoi(argv[2]);
    	//int grid_size = atoi(argv[1]);
 
    dim3 blocks( Sxo, 1, SzK ); 
    dim3 threadsPerBlock( Sxo, 1, SzI );
 
      

    	conv3D<<<threadsPerBlock,blocks>>>(d_input, d_Kernel, d_out, SxI, SzI, SxK,  SzK);   //SIZE_C1_kernel

    	cudaMemcpy(out, d_out, sizeof(float)*Sxo*Sxo*Szo, cudaMemcpyDeviceToHost);
	
	
    	cudaFree(d_Kernel);
    	cudaFree(d_input);
    	cudaFree(d_out);

	return out;

}
