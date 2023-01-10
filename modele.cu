#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include "functionMatrix.h"
#include "modele.h"

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>


#include "functionMatrix.h"
#include "dense.h"
#include "convolutionLayer.h"
#include "meanPoolingTanH.h"

#define SIZE_raw_data  6  //32
#define DEEP_raw_data  1  //32
#define SIZE_C1_kernel 3   //5
#define DEEP_K1 2//6
#define SIZE_C1_data (SIZE_raw_data-SIZE_C1_kernel+1)
#define SIZE_S1_data (SIZE_C1_data/2)        //14

#define SIZE_C2_kernel 3   //5
#define DEEP_K2 6//16
#define SIZE_C2_data (SIZE_S1_data-SIZE_C2_kernel+1)
#define SIZE_S2_data (SIZE_C2_data/2)  



#define SIZE_S3_data 120
#define SIZE_S4_data 84
#define SIZE_S5_data 10




#define TANH 0
#define SOFTMAX 1







int modele(){

    float *raw_data, *C1_data,*S1_data,*C1_kernel;
    float *C2_data,*S2_data,*C2_kernel;
    float *S3_data,*WeightD1;
    float *S4_data,*WeightD2;
    float *S5_data,*WeightD3;

    // Allocate memory
    raw_data   = init_matrix(SIZE_raw_data, SIZE_raw_data);
    C1_kernel   = init_matrix(SIZE_C1_kernel *DEEP_K1, SIZE_C1_kernel );


    
    for(int j=0; j<DEEP_K1;j++){
        
      C1_kernel[j*SIZE_C1_kernel*SIZE_C1_kernel]=1.0; 
      for(int i =1;i<SIZE_C1_kernel*SIZE_C1_kernel;i++)
      {
        C1_kernel[i+j*SIZE_C1_kernel*SIZE_C1_kernel]=0.0;
      }
    }


/*
keras.layers.Conv2D(16, kernel_size=5, strides=1, activation='tanh', padding='valid'), #C3
    keras.layers.AveragePooling2D(), #S4
    keras.layers.Flatten(), #Flatten
    keras.layers.Dense(120, activation='tanh'), #C5
    keras.layers.Dense(84, activation='tanh'), #F6
    keras.layers.Dense(10, activation='softmax') #Output layer
    #define SIZE_S3_data 120
#define SIZE_S4_data 84
#define SIZE_S5_data 10
*/

    C1_data =vectorGPUConv1(C1_kernel, raw_data,SIZE_raw_data,DEEP_raw_data,SIZE_C1_kernel,DEEP_K1 );
    S1_data =vectorGPUMeanPooling(C1_data,SIZE_C1_data,DEEP_K1);


    /*C2_data =vectorGPUConv1(C2_kernel, S1_data,SIZE_S1_data,DEEP_K1,SIZE_C2_kernel,DEEP_K2 );
    S2_data =vectorGPUMeanPooling(C2_data,SIZE_C2_data,DEEP_K2*DEEP_K1);

    vectorGPUDense (S2_data,WeightD1,S3_data,SIZE_S2_data*SIZE_S2_data*DEEP_K2*DEEP_K1,SIZE_S3_data,TANH);
    vectorGPUDense (S3_data,WeightD2,S4_data,SIZE_S3_data,SIZE_S4_data,TANH);
    vectorGPUDense (S4_data,WeightD3,S5_data,SIZE_S4_data,SIZE_S5_data,SOFTMAX);

*/






    cudaDeviceSynchronize();




    //veref
    printf("Kernel \n\n");
    print_matrix(C1_kernel,SIZE_C1_kernel*DEEP_K1,SIZE_C1_kernel);
    printf("raw_data \n\n");
    print_matrix(raw_data,SIZE_raw_data,SIZE_raw_data);
    printf("C1 \n\n");
    print_matrix(C1_data,SIZE_C1_data*DEEP_K1,SIZE_C1_data);
    printf("S1 \n\n");
     print_matrix(S1_data,SIZE_S1_data*DEEP_K1,SIZE_S1_data);


    free(raw_data);
    free(C1_data);
    free(S1_data);
    free(C1_kernel);

    return 0;
}
