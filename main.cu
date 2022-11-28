
# include <stdio.h>
//#include "functionMatrix.h"



#include <stdlib.h>
//#include <stdio.h>
//#include "functionMatrix.h"



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








float* init_matrix(int sizeX, int sizeY)
{
    float* matrice  = (float*) malloc(sizeX*sizeY*sizeof(float));
    time_t t;
    srand((unsigned) time(&t));

    for(int x=0;x<sizeX;x++){
        for(int y =0; y<sizeY; y++){
            int u = rand();
            matrice[x*sizeY+y]=((float(u)/RAND_MAX)-0.5);
            
        }
    }
    return matrice;
}


void print_matrix(float* matrice,int sizeX,int sizeY)
{

    for(int x=0;x<sizeX;x++){
        for(int y =0; y<sizeY; y++){
           
                printf(" %f ",matrice[x*sizeY+y]) ;
        }
        printf("\n");
    }
}






float* addMatrix(float* matriceA , float* matriceB, int sizeX, int sizeY)
{
     float* matrice  = (float*) malloc(sizeX*sizeY*sizeof(float));
     for(int x=0;x<sizeX;x++){
        for(int y =0; y<sizeY; y++){
           matrice[x*sizeY+y] = matriceA[x*sizeY+y] + matriceB[x*sizeY+y];
                
        }
    }
    return matrice;
}





float* multiMatrix(float* matriceA , float* matriceB, int sizeXa, int sizeYa,int sizeXb,int sizeYb)
{
    //assert(sizeXa==sizeYb);
     float* matrice  = (float*) malloc(sizeXb*sizeYa*sizeof(float));
     float u;
     for(int x=0;x<sizeXb;x++){
        for(int y =0; y<sizeYa; y++){
            u=0;
            for(int j=0;j<sizeXa;j++){
                u+=matriceA[j*sizeYa+y]*matriceB[x*sizeYb+j];
            }
           matrice[x*sizeYa+y] =u;
                
        }
    }
    return matrice;


}





int main() {
 
    c_hello();
    cuda_hello<<<1,1>>>(); 

    float* matriceA =  init_matrix(2,2);
    matriceA[0]=1;
    matriceA[1]=1;
    matriceA[2]=1;
    matriceA[3]=1;
    float* matriceB =  init_matrix(2,2);

    //float* matriceC =  multiMatrix(matriceA ,matriceA, 2,2,2,2);

    float* matriceC =  addMatrix(matriceA ,matriceA, 2,2);
    
    print_matrix(matriceC,2,2);




    cudaDeviceSynchronize();
    return 0;
}

