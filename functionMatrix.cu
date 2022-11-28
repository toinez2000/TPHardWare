
#include <stdlib.h>
#include <stdio.h>
#include "functionMatrix.h"




double* init_matrix(int sizeX, int sizeY)
{
    double* matrice  = new( double[sizeX][sizeY]);
    time_t t;
    srand((unsigned) time(&t));

    for(int x=0;x<sizeX;x++)[
        for(int y =0; y<sizeY; y++){
            int u = rand();
            if(u >= 32767/2){
                matrice[x][y]=1;
            }
            else{
                matrice[x][y]=-1;
            }

        }
    ]
}


void print_matrix(double* matrice,int sizeX,int sizeY)
{
    
    for(int x=0;x<sizeX;x++){
        for(int y =0; y<sizeY; y++){
           
                print("&fd ",matrice[x][y]) ;
        }
        print("\n");
    }
}
