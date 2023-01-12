

/**
*function print float matrix sizeX*sizeY
**/

void print_matrix(float* matrice,int sizeX,int sizeY);

/**
*function init_matrix return a float matrix sizeX*sizeY init random value between (-1,1)
**/
float* init_matrix(int sizeX, int sizeY);

/**
*function addMatrix return a float matrix sizeX*sizeY add matrice A and B use CPU
*param matriceA
*param matriceB
*return float matrix sizeX*sizeY
**/

float* addMatrix(float* matriceA , float* matriceB, int sizeX, int sizeY);

/**
*function multiMatrix return a float matrix multi matrice A and B use CPU
*param matriceA
*param matriceB
*return float matrix 
**/

float* multiMatrix(float* matriceA , float* matriceB, int sizeXa, int sizeYa,int sizeXb,int sizeYb);
