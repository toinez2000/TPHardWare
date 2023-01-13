
/**
*function vectorGPUDense Appli layer Dense with GPU
*param input   tab of the input of layer dense size SxI
*param weight  tab of the weight of the dense size SxI*Sxo+Sxo (weight+bias)
*param SxI   size input x
*param Sxo   size output x
*param ActiveFonction egal 0 ==> tanh,  1 ==> softmax
return output vector of float the result of dense layer
**/



float* vectorGPUDense (float* input, float* Weight,int SxI,int Sxo,int ActiveFunction);


//-----------------------------------------------------------------------------------------------


/**
*function Dense Appli layer Dense with GPU
*param input   tab of the input of layer dense size SxI
*param weight  tab of the weight of the dense size SxI*Sxo
*param SxI   size input x
*param Sxo   size output x
**/


__global__ void Dense(float *input,float *weight,float *output,int SxI,int Sxo);


/**
*function DTanH Appli active function with GPU
*param input   tab of the input of layer dense size SxI 
*param SxI   size input x
**/


__global__ void DTanH(float *input,int SxI);




//-------------

/**
*function softMax Appli active function with GPU
*param input   tab of the input of layer dense size SxI 
*param SxI   size input x
*param sum is the sum of the exp of each input 
**/

__global__ void softMax(float *input,int SxI,float sum);


/**
*function Expo return Expo of input matrix
*param input   tab of the input of layer dense size SxI  
*param SxI   size input x
**/



__global__ void Expo(float *input,int SxI);
