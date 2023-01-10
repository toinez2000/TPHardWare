#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "functionMatrix.h"
#include "meanPoolingTanH.h"






__device__ float activation_tanh(float M)
{
  
      M = tanh(M);
  
  
  return M; 
}

//----------------------------------------------------------------------------------------------------
//----------------------------------------------------------------------------------------------------




__global__ void meanPooling(float* input, float* output, int SxI, int SzI) {
  // Calculer les indices de l'élément de sortie courant
  int xo = blockIdx.x * blockDim.x; // ligne de l'image de sortie pour une profondeur donné 
  int yo = threadIdx.x; // element de la ligne 

  int zI = threadIdx.z ;   //profondeur du kernel à utiliser 

  // Initialiser la valeur de l'élément de sortie à 0
  float value = 0.0;

  //lien xo et xI

  
		// Appliquer le filtre sur chaque élément de l'image
	  for (int i = 0; i < 2; i++) {
	    for (int j = 0; j < 2; j++) {
	        // Calculer les indices de l'élément de l'image à utiliser
			


	        int xI = 2*xo+ i*SxI;  // lien entre xI et x0
	        int yI = 2*yo + j;
       

  		// Appliquer le filtre à l'élément de l'image courant
          value += input[xI+ yI + zI*SxI*SxI];
        }
    }



  // Enregistrer la valeur de l'élément de sortie

  
  output[xo+ yo + zI *SxI*SxI/4] = activation_tanh(value/4); // moyenne de quatre élément 
}



//-------------------------------------------------

float* vectorGPUMeanPooling(float*input,int SxI,int SzI){
    	float *out;
	    float *d_input, *d_out;

	    out = (float*)malloc(sizeof(float) *SxI*SxI*SzI/4);

	    cudaMalloc((void**)&d_input, sizeof(float)*SxI*SxI*SzI);
      cudaMalloc((void**)&d_out, sizeof(float)* SxI*SxI*SzI/4);
      cudaMemcpy(d_input, input, sizeof(float) * SxI*SxI*SzI, cudaMemcpyHostToDevice);

	// Main function
 
      dim3 blocks( SxI/2, 1, SzI ); 
      dim3 threadsPerBlock( SxI/2, 1, 1 );
 
    	meanPooling<<<threadsPerBlock,blocks>>>(d_input, d_out, SxI, SzI);

    	cudaMemcpy(out, d_out, sizeof(float)*SxI*SxI*SzI/4 , cudaMemcpyDeviceToHost);
	
	
    	cudaFree(d_input);
    	cudaFree(d_out);
	return out;

}
