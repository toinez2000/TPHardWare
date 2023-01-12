
# include <stdio.h>

#include "vector_add_N_P.h"
#include "vector_multi_N_P.h"

#include <stdlib.h>

#include "functionMatrix.h"

#include "modele.h"
#include "readfileWeight.h"
#include "affichage.h"

__global__ void cuda_hello(){
    printf("Hello World from GPU!\n");
}



void c_hello(){
    printf("Hello World!\n");
}







/* int main() {
    c_hello();
    return 0;
}
*/


int main() {
 

//readfile();


 /*
    c_hello();
    cuda_hello<<<1,1>>>(); 

    float* matriceA =  init_matrix(1000,1000);
    matriceA[0]=1;
    matriceA[1]=2;
    matriceA[2]=1;
    matriceA[3]=1;
    float* matriceB =  init_matrix(2,2);

    float* matriceC =  multiMatrix(matriceA ,matriceA, 1000,1000,1000,1000);

    //float* matriceC =  addMatrix(matriceA ,matriceA, 2,2);


//float* matriceC = vectorGPUAdd (matriceA, matriceA, 2, 2);
//float* matriceC = multimatriceGPU(matriceA, matriceA, 1000);

    
    
    

*/

    mainAffiche();
    float* matriceinput =  readImage();


printf("BeginMain \n");

/*
    print_matrix(matriceinput,32,32);*/

    float*output =  modele(matriceinput);
    //cudaDeviceSynchronize();
    //print_matrix(matriceC,10,10);

    printf("output \n");
    print_matrix(output,10,1);




    return 0;
}

// Pour multiplication GPUsur N=1000 time executiuon = 0.236s
// Pour multiplication CPU sur N=1000 time executiuon = 8.650s
