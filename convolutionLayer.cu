#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "functionMatrix.h"
#include "convolutionLayer.h"
#include "vector_add_N_P.h"

<<<<<<< HEAD
=======
__global__ void addLayerOutput(float* input, float* output, int SxI, int SzI, int SxK, int SzK) {
  // Calculer les indices de l'élément de sortie courant
  int xo = blockIdx.x * blockDim.x; // ligne de l'image de sortie pour une Deep input et output donnée
  int yo = threadIdx.x; // element de la ligne pour une Deep input et output donnée
  int zK = blockIdx.z ; // Deep du kernel  
  int Sxo = (SxI-SxK+1); // Size output en x

	float value =0;
  
		// Appliquer le filtre sur chaque élément de l'image
	  for (int i = 0; i < SzI; i++) {
          	
           value += input[xo+yo+  zK*Sxo*Sxo + i*SzK*Sxo*Sxo];  // sum(Cx_data[x_c= const][y_c = const ][zK = const][zI]) 
	      
	    }

  	  // Enregistrer la valeur dans la sortie
    output[xo+ yo + zK*Sxo*Sxo] = value;
         
}




>>>>>>> dvt
__global__ void conv3D(float* input, float* kernel, float* output, int SxI, int SzI, int SxK, int SzK) {
  // Calculer les indices de l'élément de sortie courant
  int xo = blockIdx.x * blockDim.x; // ligne de l'image de sortie pour une profondeur donnée 
  int yo = threadIdx.x; // element de la ligne
  int Sxo = (SxI-SxK+1); //Size output x
  int Szo = SzK; //Deep output 

 

  int zK = blockIdx.z ;   //profondeur du kernel à utiliser 
  int zI = threadIdx.z; // proffondeur de l'image input à utiliser

  // Initialiser la valeur de l'élément de sortie à 0
  float value = 0.0;



  
		// Appliquer le filtre sur chaque élément de l'image
	  for (int i = 0; i < SxK; i++) {
	    for (int j = 0; j < SxK; j++) {
	        // Calculer les indices de l'élément de l'image à utiliser
			

			int decalage = blockIdx.x*(SxK-1);


<<<<<<< HEAD
          int xI = xo+ i*SxI + decalage +zI*(SxI*SxI);  // lien entre xI et x0
	        int yI = yo + j;
          	// Appliquer le filtre à l'élément de l'image courant
           value += input[xI+ yI] * kernel[i*SxK + j + zK*SxK*SxK+ zI*SzK*SxK*SxK];
          //printf("xo = %d,yo = %d, zI = %d, zk =%d , indiceInput = %d indiceKernel= %d \n",xo,yo,zI,zK,xI+ yI,i*SxK + j + zK*SxK*SxK);
=======
          int xI = xo+ i*SxI + decalage ;  // lien entre xI et x0
	        int yI = yo + j;
          	// Appliquer le filtre à l'élément de l'image courant
           value += input[xI+ yI+zI*(SxI*SxI)] * kernel[i*SxK + j + zK*SxK*SxK+ zI*SzK*SxK*SxK];
          //printf("xo = %d,yo = %d, zI = %d, zk =%d , indiceInput = %d indiceKernel= %d \n",xo,yo,zI,zK,xI+ yI,i*SxK + j + zK*SxK*SxK + zI*SzK*SxK*SxK);
>>>>>>> dvt
          //printf("xo = %f ,indiceK = %d \n",kernel[i*SxK + j + zK*SxK*SxK],i*SxK + j + zK*SxK*SxK);
	      }
	    }

  	  // Enregistrer la valeur
    output[xo+ yo + zK*Sxo*Sxo + zI*Szo*Sxo*Sxo] = value;     
}


__global__ void addBias(float* M, float*bias, int SxI, int SzI) {
  // Calculer les indices de l'élément de sortie courant
  int deep = blockIdx.x; //  profondeur
  int yo = threadIdx.x; // element 
 

<<<<<<< HEAD
     
    output[xo*SxI*SxI+ yo] += bias[deep];
=======
     //printf("bias = %f ", bias[deep]);
    M[deep*SxI*SxI+ yo] += bias[deep];
>>>>>>> dvt
  
         
}





float* vectorGPUConv1 (float* Kernel, float* input,int SxI,int SzI,int SxK,int SzK)
{

<<<<<<< HEAD
	float *out;
	float *d_input, *d_Kernel, *d_out;
  int Sxo = SxI-SxK+1;
  int Szo = SzK;

	out = (float*)malloc(sizeof(float) * Sxo*Sxo*Szo);

	cudaMalloc((void**)&d_input, sizeof(float)*SxI*SxI*SzI);
    	cudaMalloc((void**)&d_Kernel, sizeof(float)*SxK*SxK*SzK*SzI+SzK);
    	cudaMalloc((void**)&d_out, sizeof(float)*Sxo*Sxo*Szo);

	
    	cudaMemcpy(d_Kernel, Kernel, sizeof(float) *SxK*SxK*SzK*SzI+SzK, cudaMemcpyHostToDevice);
=======
	float *out ;
	float *d_input, *d_Kernel, *d_out, *d_out0;  
  	int Sxo = SxI-SxK+1;  //size output x convolution layer
  	int Szo = SzK;


	out = (float*)malloc(sizeof(float) * Sxo*Sxo*Szo);
/*
	printf("C1 \n\n");
    print_matrix(out,Sxo*Szo*SzI,Sxo);
*/

	//réservation de mémoire CPU
	cudaMalloc((void**)&d_input, sizeof(float)*SxI*SxI*SzI);  
    	cudaMalloc((void**)&d_Kernel, sizeof(float)*(SxK*SxK*SzK*SzI+SzK));
    	cudaMalloc((void**)&d_out, sizeof(float)*Sxo*Sxo*Szo);
	cudaMalloc((void**)&d_out0, sizeof(float)*Sxo*Sxo*Szo*SzI);  //d_out0 output intermédiaire cette matrice contient l'output de la convolution Sxo*Sxo*Szo*SzI

	
    	cudaMemcpy(d_Kernel, Kernel, sizeof(float) *(SxK*SxK*SzK*SzI+SzK), cudaMemcpyHostToDevice);
>>>>>>> dvt
    	cudaMemcpy(d_input, input, sizeof(float) *SxI*SxI*SzI, cudaMemcpyHostToDevice);

	// Main function
    	//int block_size = atoi(argv[2]);
    	//int grid_size = atoi(argv[1]);
 
<<<<<<< HEAD
    dim3 blocks( Sxo, 1, SzK ); 
    dim3 threadsPerBlock( Sxo, 1, SzI );
 
      

    	conv3D<<<threadsPerBlock,blocks>>>(d_input, d_Kernel, d_out, SxI, SzI, SxK,  SzK);   //SIZE_C1_kernel
	addBias<<<Sxo*Sxo,SzK>>>(d_out, d_Kernel+SxK*SxK*SzI*SzK, Sxo, SzK);
	
=======
    dim3 blocks( Sxo, 1, SzI ); 
    dim3 threadsPerBlock( Sxo, 1, SzK );
 
      

    	conv3D<<<threadsPerBlock,blocks>>>(d_input, d_Kernel, d_out0, SxI, SzI, SxK,  SzK);  
		cudaDeviceSynchronize();
		
		//printf("SzI = %d \n",SzI);
		if(SzI>1){

			dim3 blocks1( Sxo, 1, 1); 
			addLayerOutput<<<threadsPerBlock,blocks1>>>(d_out0, d_out,SxI, SzI,  SxK,  SzK);
			cudaDeviceSynchronize();
		
			addBias<<<SzK,Sxo*Sxo>>>(d_out, d_Kernel+SxK*SxK*SzI*SzK, Sxo, SzK);
			cudaDeviceSynchronize();
    		cudaMemcpy(out, d_out, sizeof(float)*Sxo*Sxo*Szo, cudaMemcpyDeviceToHost);
			}
		else{
			addBias<<<SzK,Sxo*Sxo>>>(d_out, d_Kernel+SxK*SxK*SzI*SzK, Sxo, SzK);
			cudaDeviceSynchronize();
    		cudaMemcpy(out, d_out0, sizeof(float)*Sxo*Sxo*Szo, cudaMemcpyDeviceToHost);
		}
	
	//free
    	cudaFree(d_Kernel);
    	cudaFree(d_input);
    	cudaFree(d_out);
	cudaFree(d_out0);

	return out;

}
