#include <wb.h>

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "CUDA error: ", cudaGetErrorString(err));              \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      return -1;                                                          \
    }                                                                     \
  } while (0)

//@@ Define any useful program-wide constants here

#define Mask_Width 3
#define Mask_Radius 1
#define Tile_Width 4
#define Block_Width Tile_Width+Mask_Width-1

//@@ Define constant memory for device kernel here

__constant__ float Mask [Mask_Width][Mask_Width][Mask_Width];


__global__ void conv3d(float *input, float *output, const int z_size,
                       const int y_size, const int x_size) {
  
  //@@ Insert kernel code here
  
  __shared__ float N_ds[Block_Width][Block_Width][Block_Width];
  int bx = blockIdx.x;
  int by = blockIdx.y;
  int bz = blockIdx.z;
 
  int tx = threadIdx.x;
  int ty = threadIdx.y;
  int tz = threadIdx.z;
  
  int x_o = bx * Tile_Width + tx;
  int y_o = by * Tile_Width + ty;
  int z_o = bz * Tile_Width + tz;
  
  int x_i = x_o - Mask_Radius;
  int y_i = y_o - Mask_Radius;
  int z_i = z_o - Mask_Radius;
  
  if ((x_i >= 0) && (x_i < x_size) && (y_i >= 0) && (y_i<y_size) && (z_i>=0) && (z_i<z_size)){
    N_ds[tz][ty][tx] = input[z_i * x_size * y_size + y_i * x_size + x_i];
  }else{
    N_ds[tz][ty][tx] = 0.0f;
  }
  __syncthreads();
  
  float result = 0.0f;
  if (tx<Tile_Width&&ty<Tile_Width&&tz<Tile_Width){
    for (int i=0;i<Mask_Width;i++){
      for (int j=0;j<Mask_Width;j++){
        for (int k=0;k<Mask_Width;k++){
          //if ((x_o >= 0) && (x_o < x_size) && (y_o >= 0) && (y_o < y_size) && (z_o >= 0) && (z_o < z_size)){
          result += Mask[i][j][k]*N_ds[i+tz][j+ty][k+tx];
          //}
        }
      }
    }
      if ((x_o<x_size)&&(y_o<y_size)&&(z_o<z_size)){
    output[z_o*x_size*y_size+ y_o*x_size+x_o] = result;
    }
  }
  __syncthreads();
}



int main(int argc, char *argv[]) {
  wbArg_t args;
  int z_size;
  int y_size;
  int x_size;
  int inputLength, kernelLength;
  float *hostInput;
  float *hostKernel;
  float *hostOutput;
  float *deviceInput;
  float *deviceOutput;

  args = wbArg_read(argc, argv);

  // Import data
  hostInput = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);
  hostKernel =
      (float *)wbImport(wbArg_getInputFile(args, 1), &kernelLength);
  hostOutput = (float *)malloc(inputLength * sizeof(float));

  // First three elements are the input dimensions
  z_size = hostInput[0];
  y_size = hostInput[1];
  x_size = hostInput[2];
  wbLog(TRACE, "The input size is ", z_size, "x", y_size, "x", x_size);
  assert(z_size * y_size * x_size == inputLength - 3);
  assert(kernelLength == 27);

  wbTime_start(GPU, "Doing GPU Computation (memory + compute)");

  wbTime_start(GPU, "Doing GPU memory allocation");
  //@@ Allocate GPU memory here
  // Recall that inputLength is 3 elements longer than the input data
  // because the first  three elements were the dimensions
  
  cudaMalloc((void**)&deviceInput, x_size*y_size*z_size*sizeof(float));
  cudaMalloc((void**)&deviceOutput, x_size*y_size*z_size*sizeof(float));
  
  wbTime_stop(GPU, "Doing GPU memory allocation");

  wbTime_start(Copy, "Copying data to the GPU");
  //@@ Copy input and kernel to GPU here
  // Recall that the first three elements of hostInput are dimensions and
  // do
  // not need to be copied to the gpu
  
  cudaMemcpy(deviceInput, &hostInput[3], z_size*y_size*x_size*sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpyToSymbol(Mask, hostKernel, Mask_Width*Mask_Width*Mask_Width*sizeof(float));
  
  
  wbTime_stop(Copy, "Copying data to the GPU");

  wbTime_start(Compute, "Doing the computation on the GPU");
  //@@ Initialize grid and block dimensions here
  
  dim3 DimGrid(ceil(x_size/(1.0*Tile_Width)),ceil(y_size/(1.0*Tile_Width)),ceil(z_size/(1.0*Tile_Width)));
  dim3 DimBlock(Block_Width, Block_Width, Block_Width);

  //@@ Launch the GPU kernel here
  
  conv3d <<<DimGrid,DimBlock>>>(deviceInput, deviceOutput, z_size, y_size, x_size);
  
  
  cudaDeviceSynchronize();
  wbTime_stop(Compute, "Doing the computation on the GPU");

  wbTime_start(Copy, "Copying data from the GPU");
  //@@ Copy the device memory back to the host here
  // Recall that the first three elements of the output are the dimensions
  // and should not be set here (they are set below)
  
  cudaMemcpy(&hostOutput[3], deviceOutput, x_size*y_size*z_size*sizeof(float), cudaMemcpyDeviceToHost);
  
  wbTime_stop(Copy, "Copying data from the GPU");

  wbTime_stop(GPU, "Doing GPU Computation (memory + compute)");

  // Set the output dimensions for correctness checking
  hostOutput[0] = z_size;
  hostOutput[1] = y_size;
  hostOutput[2] = x_size;
  wbSolution(args, hostOutput, inputLength);

  // Free device memory
  cudaFree(deviceInput);
  cudaFree(deviceOutput);

  // Free host memory
  free(hostInput);
  free(hostOutput);
  return 0;
}