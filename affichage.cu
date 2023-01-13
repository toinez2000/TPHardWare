
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define WIDTH 28
#define HEIGHT 28


void charBckgrndPrint(char *str, int rgb[3]){
  printf("\033[48;2;%d;%d;%dm", rgb[0], rgb[1], rgb[2]);
  printf("%s\033[0m",str);
}

void imgColorPrint(int height, int width, int ***img){
  int row, col;
  char *str="  ";
  for(row=0; row<height; row++){
    for(col=0; col<width; col++){
      charBckgrndPrint(str,img[row][col]);
    }
    printf("\n");
  }
}

int mainAffiche() {
  int i, j;
  int ***img;
  int color[3]={255,0,0};
  unsigned int magic, nbImg, nbRows, nbCols;
  unsigned char val;
  FILE *fptr;

  // Malloc image
  img = (int ***)malloc(HEIGHT*sizeof(int **));
  for(i=0; i<HEIGHT; i++){
    img[i]= (int **)malloc(WIDTH*sizeof(int *));
    for(j=0; j<WIDTH; j++){
      img[i][j] = (int *)malloc(sizeof(int)*3);
    }
  }

  //Open File
  if((fptr = fopen("train-images.idx3-ubyte","rb")) == NULL){
    printf("Can't open file");
    exit(1);
  }

  //Read File
  fread(&magic, sizeof(int), 1, fptr);
  fread(&nbImg, sizeof(int), 1, fptr);
  fread(&nbRows, sizeof(int), 1, fptr);
  fread(&nbCols, sizeof(int), 1, fptr);
/*
  printf("Nb Magic : %u \n", magic);
  printf("Nb Img : %u \n", nbImg);
  printf("Nb Rows : %u \n", nbRows);
  printf("Nb Cols : %u \n", nbCols);
*/
  for(i=0; i<HEIGHT; i++){
    for(j=0; j<WIDTH; j++){ 
      fread(&val, sizeof(unsigned char), 1, fptr);  
      img[i][j][0]=(int)val*color[0]/255;
      img[i][j][1]=(int)val*color[1]/255;
      img[i][j][2]=(int)val*color[2]/255;
    }
  }

  imgColorPrint(HEIGHT, WIDTH, img);

  // setup image grayscale
  for(i=0; i<HEIGHT; i++){
    for(j=0; j<WIDTH; j++){
        img[i][j][0] = ((i+j)*4)%255;
        img[i][j][1] = ((i+j)*4)%255;
        img[i][j][2] = ((i+j)*4)%255;
    }
  }

  // print image
  imgColorPrint(HEIGHT, WIDTH, img);
  return 0;
  //exit(EXIT_SUCCESS);
}



float* readImage() {

  printf("read input mnist \n");
  int i, j;
  float* output;
  int ***img;
  int color[3]={255,0,0};
  unsigned int magic, nbImg, nbRows, nbCols;
  unsigned char val;
  FILE *fptr;


 

  //Open File
  if((fptr = fopen("train-images.idx3-ubyte","rb")) == NULL){
    printf("Can't open file");
    exit(1);
  }

  //Read File
  fread(&magic, sizeof(int), 1, fptr);
  fread(&nbImg, sizeof(int), 1, fptr);
  fread(&nbRows, sizeof(int), 1, fptr);
  fread(&nbCols, sizeof(int), 1, fptr);
/*
  printf("Nb Magic : %u \n", magic);
  printf("Nb Img : %u \n", nbImg);
  printf("Nb Rows : %u \n", nbRows);
  printf("Nb Cols : %u \n", nbCols);
*/

 output = (float *)malloc(sizeof(float)*32*32);
  float max =0;

  //flatten img and reshape 32*32
 
  for(int x=0;x<28;x++)
        {
            for(int y=0; y<28;y++){
                fread(&val, sizeof(unsigned char), 1, fptr);
                output[x+y*28] = (int)val*color[0]/255;

                if (max<output[x+y*28])
                { max = output[x+y*28];}
            }
        }
        for(int z=0;z<240;z++){
            output[28*28+z] = 0;

        }

/*
        for(int x=0;x<32*32;x++){
          output[x]= ((output[x]/max)-0.5)*2; //normalize
        }
*/
  
  return output;
}
