#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "functionMatrix.h"


#define SIZE_raw_data  4  //32
#define SIZE_C1_kernel 2   //5
#define DEEP 2//6
#define SIZE_C1_data (SIZE_raw_data-SIZE_C1_kernel+1)
#define SIZE_S1_data 14







__global__ void convolutionLayer(float* C1_data,float* C1_kernel, float* raw_data){

    int tind = blockIdx.x*blockDim.x+threadIdx.x;
    // tid = ligne*tailleLigne+elementLigne
    
    int deepN = ((int)tind)/((int)(SIZE_C1_data*SIZE_C1_data));
    //printf("deepN = %d,tind = %d, SIZE_C1_data = %d \n",deepN,tind,SIZE_C1_data*SIZE_C1_data);

    int a = tind+(SIZE_C1_kernel-1)*(blockIdx.x);


    for(int i=0;i< SIZE_C1_kernel;i++){
    for(int j=0; j < SIZE_C1_kernel; j++){


            
            
           
          

            
            C1_data[tind]+= raw_data[a+i*SIZE_raw_data+j-deepN*(SIZE_raw_data*SIZE_raw_data)]*C1_kernel[(i*SIZE_C1_kernel+j)+deepN*(SIZE_C1_kernel*SIZE_C1_kernel)];
            //printf("%f \n",C1_data[tind]);
            
            
            if(tind == 5 && i == 0 && j ==0){
            printf("a = %d \n",a);
            printf("deepN = %d \n",deepN);
            printf("SIZE_C1_data  = %d \n",SIZE_C1_data );
            printf("%f \n",raw_data[a+i*SIZE_raw_data+j-deepN*(SIZE_raw_data*SIZE_raw_data)]);
            }


        }
    }
    //printf("%d \n",tind);

   


}

float* vectorGPUConv1 (float* Kernel, float* input)
{

	float *out;
	float *d_input, *d_Kernel, *d_out;

	out = (float*)malloc(sizeof(float) * SIZE_C1_data*SIZE_C1_data*DEEP);

	cudaMalloc((void**)&d_input, sizeof(float)*SIZE_raw_data*SIZE_raw_data);
    	cudaMalloc((void**)&d_Kernel, sizeof(float)*SIZE_C1_kernel*SIZE_C1_kernel*DEEP);
    	cudaMalloc((void**)&d_out, sizeof(float)*SIZE_C1_data *SIZE_C1_data*DEEP);

	
    	cudaMemcpy(d_Kernel, Kernel, sizeof(float) * SIZE_C1_kernel*SIZE_C1_kernel*DEEP, cudaMemcpyHostToDevice);
    	cudaMemcpy(d_input, input, sizeof(float) * SIZE_raw_data*SIZE_raw_data, cudaMemcpyHostToDevice);

	// Main function
    	//int block_size = atoi(argv[2]);
    	//int grid_size = atoi(argv[1]);
	int block_size = DEEP*SIZE_C1_data ;
    	int grid_size =SIZE_C1_data ;
    	convolutionLayer<<<grid_size,block_size>>>(d_out,d_Kernel, d_input);

    	cudaMemcpy(out, d_out, sizeof(float)*SIZE_C1_data *SIZE_C1_data*DEEP , cudaMemcpyDeviceToHost);
	
	
    	cudaFree(d_Kernel);
    	cudaFree(d_input);
    	cudaFree(d_out);

	return out;

}


int main2(){

    float *raw_data, *C1_data,*S1_data,*C1_kernel;


    // Allocate memory
    raw_data   = init_matrix(SIZE_raw_data, SIZE_raw_data);
    C1_data   = (float*)malloc(sizeof(float) * DEEP*SIZE_C1_data*SIZE_C1_data);
    S1_data   = (float*)malloc(sizeof(float) * DEEP*SIZE_S1_data*SIZE_S1_data);
    C1_kernel   = init_matrix(SIZE_C1_kernel *DEEP, SIZE_C1_kernel );
    C1_kernel[0]=1; 
    C1_kernel[1]=0;
    C1_kernel[2]=0;
    C1_kernel[3]=0;

    C1_kernel[4]=1; 
    C1_kernel[5]=0;
    C1_kernel[6]=0;
    C1_kernel[7]=0;



    




//Une matrice float raw_data de taille 32x32 initialisé avec des valeurs comprises entre 0 et 1, correspondant à nos données d'entrée.
//Une matrice float C1_data de taille 6x28x28 initialisé à 0 qui prendra les valeurs de sortie de la convolution 2D. C1 correspond aux données après la première Convolution.
//Une matrice float S1_data de taille 6x14x14 intialisé à 0 qui prendra les valeurs de sortie du sous-échantillonnage. S1 correspond aux données après le premier Sous-échantillonnage.
//Une matrice float C1_kernel de taille 6x5x5 initialisé à des valeurs comprises entre 0 et 1 correspondant à nos premiers noyaux de convolution.





    float* matriceC =vectorGPUConv1(C1_kernel, raw_data );
    cudaDeviceSynchronize();
    print_matrix(raw_data,4,4);
    printf("\n");
    print_matrix(matriceC,SIZE_C1_data*DEEP,SIZE_C1_data);


    free(raw_data);
    free(C1_data);
    free(S1_data);
    free(C1_kernel);
    free(matriceC);

    return 0;
}



