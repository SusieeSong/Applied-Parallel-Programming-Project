 // MP Scan
// Given a list (lst) of length n
// Output its prefix sum = {lst[0], lst[0] + lst[1], lst[0] + lst[1] + ...
// +
// lst[n-1]}

#include <wb.h>

#define BLOCK_SIZE 512 //@@ You can change this

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

__global__ void scan(float *input, float *output, int len) {
  //@@ Modify the body of this function to complete the functionality of
  //@@ the scan on the device
  //@@ You may need multiple kernel calls; write your kernels before this 
  //@@ function and call them from the host
  
 __shared__ float XY[2*BLOCK_SIZE];
  unsigned int t = threadIdx.x;
  unsigned int start = 2*blockIdx.x*blockDim.x;
  if (start+t < len){
    XY[t] = input[start + t];
  }
  else{
    XY[t] = 0;
  }
  if ((start+t+blockDim.x) < len){
    XY[blockDim.x+t] = input[start+blockDim.x+t];
  }
  else{
    XY[blockDim.x+t] = 0;
  }
  
  for (unsigned int stride = 1; stride <= blockDim.x; stride*=2){
    __syncthreads();
    int index = (threadIdx.x+1)*2*stride-1;
    if ( index < (2*BLOCK_SIZE)){
      XY[index]+= XY[index-stride];
    }
  }
  
  for (int stride= (BLOCK_SIZE*2)/4; stride>0; stride/=2){
    __syncthreads();
    int index = (threadIdx.x+1)*stride*2-1;
    if((index+stride)<(BLOCK_SIZE*2)){
      XY[index+stride]+=XY[index];
    }
  }
  
  __syncthreads();
 if(start+t<len) 
    output[start+t]=XY[threadIdx.x];
  if(start+t+blockDim.x<len)
    output[start+t+blockDim.x]= XY[threadIdx.x+blockDim.x];    
}

__global__ void finaladd(float *in_d, float *out_d, int len) { 
  __shared__ float XY[2*BLOCK_SIZE];
  
  int b = blockIdx.x; int t = threadIdx.x;
  int index = b * blockDim.x + t;
  if (index<len) XY[t] = in_d[index]; 
  float temp = 0.0;
  if (b > 0) for (int i = 1; i <= b; i++) temp += in_d[i * 2 * BLOCK_SIZE - 1];
  XY[t] += temp;
  if (index<len) out_d[index] = XY[t];
}



int main(int argc, char **argv) {
  wbArg_t args;
  float *hostInput;  // The input 1D list
  float *hostOutput; // The output list
  float *deviceInput;
  float *deviceOutput;
  int numElements; // number of elements in the list

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &numElements);
  hostOutput = (float *)malloc(numElements * sizeof(float));
  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The number of input elements in the input is ",
        numElements);

  wbTime_start(GPU, "Allocating GPU memory.");
  wbCheck(cudaMalloc((void **)&deviceInput, numElements * sizeof(float)));
  wbCheck(cudaMalloc((void **)&deviceOutput, numElements * sizeof(float)));
  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Clearing output memory.");
  wbCheck(cudaMemset(deviceOutput, 0, numElements * sizeof(float)));
  wbTime_stop(GPU, "Clearing output memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  wbCheck(cudaMemcpy(deviceInput, hostInput, numElements * sizeof(float),
                     cudaMemcpyHostToDevice));
  wbTime_stop(GPU, "Copying input memory to the GPU.");

//@@ Initialize the grid and block dimensions here
  
    dim3 DimGrid(numElements/(2*BLOCK_SIZE), 1, 1);
    if (numElements%(2*BLOCK_SIZE)) DimGrid.x++;
    dim3 DimBlock(BLOCK_SIZE, 1, 1);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Modify this to complete the functionality of the scan
  //@@ on the deivce
   scan<<<DimGrid, DimBlock>>>(deviceInput, deviceOutput,numElements);
   DimBlock.x = 2*BLOCK_SIZE; 
   finaladd<<<DimGrid, DimBlock>>>(deviceOutput, deviceInput, numElements); 
  

  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Performing CUDA computation");

  wbTime_start(Copy, "Copying output memory to the CPU");
  wbCheck(cudaMemcpy(hostOutput, deviceInput, numElements * sizeof(float),
                     cudaMemcpyDeviceToHost));
  wbTime_stop(Copy, "Copying output memory to the CPU");

  wbTime_start(GPU, "Freeing GPU Memory");
  cudaFree(deviceInput);
  cudaFree(deviceOutput);
  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostOutput, numElements);

  free(hostInput);
  free(hostOutput);

  return 0;
}
