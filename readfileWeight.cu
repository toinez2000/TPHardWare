
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>


float* readfile() {
  float*WeightVector;
 
  float val;
  FILE *fptr;

  // Malloc vectorWeight



    int SizeWeight = (5*5*6+6)+(5*5*6*16+16)+(400*120+120)+(84*120+84)+(84*10+10);

  WeightVector = (float*)malloc(SizeWeight*sizeof(float));


  //Open File
  if((fptr = fopen("valeurs.dat","rb")) == NULL){
    printf("Can't open file");
    exit(1);
  }

  //Read File
  


  for(int i=0; i<SizeWeight; i++){
    fread(&val, sizeof(float), 1, fptr);
    WeightVector[i] = val;
    //printf("%f \n",val);
    }

    return WeightVector;
  
}