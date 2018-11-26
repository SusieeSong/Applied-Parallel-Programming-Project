// Histogram Equalization

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

#define HISTOGRAM_LENGTH 256
#define BLOCK_SIZE 32

//@@ insert code here

__global__ void floattochar(float *inputImage, unsigned char *outputImage, int width, int height, int channels){
  
  int Col = blockIdx.x * blockDim.x + threadIdx.x; 
  int Row = blockIdx.y * blockDim.y + threadIdx.y;
  if (Col < width && Row < height){
    int i = (width * Row + Col) * channels;

  outputImage[i] = (unsigned char) (255 * inputImage[i]);
  outputImage[i + 1] = (unsigned char) (255 * inputImage[i + 1]);
  outputImage[i + 2] = (unsigned char) (255 * inputImage[i + 2]);
  }
}


__global__ void rgbtogrey(unsigned char *inputImage, unsigned char *outputImage, int width, int height, int channels){
  
  int Col = blockIdx.x * blockDim.x + threadIdx.x; 
  int Row = blockIdx.y * blockDim.y + threadIdx.y;
  int i = (width * Row + Col) * channels;

  unsigned char r = inputImage[i];
  unsigned char g = inputImage[i + 1];
  unsigned char b = inputImage[i + 2];
  
  if (Col < width && Row < height){
    outputImage[width * Row + Col] = (unsigned char) (0.21*r + 0.71*g + 0.07*b);
  }
}



__global__ void histogram1(unsigned char *inputImage, float *hist,  int width, int height){
  
  int i = blockIdx.x * blockDim.x + threadIdx.x; 
  int stride = blockDim.x* gridDim.x;
  while(i < (width * height)){
  atomicAdd( &(hist[inputImage[i]]), 1/(width*height*1.0));
        i = i + stride;
       }
  __syncthreads(); 
}


__global__ void histogram2(unsigned char *inputImage, float *hist,  int width, int height){

  int i = threadIdx.x + blockIdx.x * blockDim.x;
  
  __shared__ float priviate_histo[256];
  if(threadIdx.x < 256) priviate_histo[threadIdx.x] = 0.0;
  __syncthreads();
  
  int stride = blockDim.x* gridDim.x;
  while (i < width*height){
    atomicAdd(&(priviate_histo[inputImage[i]]),1.0);
    i+=stride;
  }
  __syncthreads();
  if(threadIdx.x < 256){
    atomicAdd(&hist[threadIdx.x], priviate_histo[threadIdx.x]/(width*height*1.0));
  }
}

__global__ void scan1(float *input, float *output, int len) {
    
  __shared__ float sum[HISTOGRAM_LENGTH];

  int bx = blockIdx.x; int tx = threadIdx.x;
  int start = bx * blockDim.x * 2 + tx;

  if (start < len) sum[tx] = input[start];
  else sum[tx] = 0;

  if (start + blockDim.x < len) sum[tx + blockDim.x] = input[start + blockDim.x];
  else sum[tx + blockDim.x] = 0;
  
  __syncthreads();

  for (int stride = 1; stride <= blockDim.x; stride *= 2)
  {
    int i = (tx+1) * stride * 2 - 1; 
    if(i < 2 * blockDim.x) sum[i] += sum[i-stride];
    
    __syncthreads();
  }

  for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
  {
    int i = (tx+1) * stride * 2 - 1;
    if(i + stride < 2 * blockDim.x) sum[i + stride] += sum[i]; 
    
    __syncthreads();
  }

  if (start < len) output[start] = sum[tx];
  if (start + blockDim.x < len) output[start+blockDim.x] = sum[tx+blockDim.x];

}


__global__ void scan2(float *input, float *output, int len) {
  
 __shared__ float sum[HISTOGRAM_LENGTH];

  int bx = blockIdx.x; int tx = threadIdx.x;
  int start = bx * blockDim.x + tx;

  if (start < len) sum[tx] = input[start];
  //else sum[tx] = 0;

  //if (start + blockDim.x < len) sum[tx + blockDim.x] = input[start + blockDim.x];
 // else sum[tx + blockDim.x] = 0;
  
 for (int stride = 1; stride <= blockDim.x; stride *= 2)
  {
      __syncthreads(); 
    if (tx >= stride)
      sum[tx] += sum[tx - stride];
  }
  output[start] = sum[tx];
}





__global__ void cdf(unsigned char *image, float *histcdf, int width, int height, int channels){
  
  int Col = blockIdx.x * blockDim.x + threadIdx.x; 
  int Row = blockIdx.y * blockDim.y + threadIdx.y;
  if (Col < width && Row < height){
  int index = (width * Row + Col) * channels;

  for (int i=0; i<3; i++){
    image[index+i] = (unsigned char)(min(max(255.0*(histcdf[image[index+i]] - histcdf[0])/(1 - histcdf[0]), 0.0), 255.0));
  }
  }
}


__global__ void chartofloat(unsigned char *inputImage, float *outputImage, int width, int height, int channels){
  
  int Col = blockIdx.x * blockDim.x + threadIdx.x; 
  int Row = blockIdx.y * blockDim.y + threadIdx.y;
  if (Col < width && Row < height){
  int index = (width * Row + Col) * channels;

  outputImage[index] = (float)(inputImage[index])/255.0;
  outputImage[index + 1] = (float)(inputImage[index + 1])/255.0;
  outputImage[index + 2] = (float)(inputImage[index + 2])/255.0;

  }
}






int main(int argc, char **argv) {
  wbArg_t args;
  int imageWidth;
  int imageHeight;
  int imageChannels;
  wbImage_t inputImage;
  wbImage_t outputImage;
  float *hostInputImageData;
  //float *hostOutputImageData;
  const char *inputImageFile;

  //@@ Insert more code here
  
  float *deviceInputImage;
  unsigned char *deviceColorImage;
  unsigned char *deviceGreyImage;
  float *deviceHistogram;
  float *deviceHistogramCdf;

  args = wbArg_read(argc, argv); /* parse the input arguments */

  inputImageFile = wbArg_getInputFile(args, 0);

  wbTime_start(Generic, "Importing data and creating memory on host");
  inputImage = wbImport(inputImageFile);
  imageWidth = wbImage_getWidth(inputImage);
  imageHeight = wbImage_getHeight(inputImage);
  imageChannels = wbImage_getChannels(inputImage);
  outputImage = wbImage_new(imageWidth, imageHeight, imageChannels);
  wbTime_stop(Generic, "Importing data and creating memory on host");

  //@@ insert code here
  
  hostInputImageData = wbImage_getData(inputImage);
  

  wbCheck(cudaMalloc((void **) &deviceInputImage, imageWidth * imageHeight * imageChannels * sizeof(float))); 
  wbCheck(cudaMalloc((void **) &deviceColorImage, imageWidth * imageHeight * imageChannels * sizeof(unsigned char))); 
  wbCheck(cudaMalloc((void **) &deviceGreyImage, imageWidth * imageHeight * sizeof(unsigned char)));
  wbCheck(cudaMalloc((void **) &deviceHistogram, HISTOGRAM_LENGTH * sizeof(float))); 
  wbCheck(cudaMalloc((void **) &deviceHistogramCdf, HISTOGRAM_LENGTH * sizeof(float))); 
  

  wbCheck(cudaMemcpy(deviceInputImage, hostInputImageData,imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyHostToDevice)); // copy the variables into gpu memory
   
  dim3 dimGrid(imageWidth/BLOCK_SIZE,imageHeight/BLOCK_SIZE,1);
  if (imageWidth%BLOCK_SIZE) dimGrid.x++;
  if (imageHeight%BLOCK_SIZE) dimGrid.y++;
  dim3 dimBlock(BLOCK_SIZE,BLOCK_SIZE,1);
  

  floattochar<<<dimGrid, dimBlock>>>(deviceInputImage, deviceColorImage, imageWidth, imageHeight, imageChannels);
  rgbtogrey<<<dimGrid, dimBlock>>>(deviceColorImage, deviceGreyImage, imageWidth, imageHeight, imageChannels);
  histogram1<<<32, 256>>>(deviceGreyImage, deviceHistogram, imageWidth, imageHeight);
  scan1<<<1, HISTOGRAM_LENGTH/2>>>(deviceHistogram, deviceHistogramCdf, HISTOGRAM_LENGTH);
  //scan2<<<1, HISTOGRAM_LENGTH>>>(deviceHistogram, deviceHistogramCdf, HISTOGRAM_LENGTH);
  cdf<<<dimGrid, dimBlock>>>(deviceColorImage, deviceHistogramCdf, imageWidth, imageHeight, imageChannels);
  chartofloat<<<dimGrid, dimBlock>>>(deviceColorImage, deviceInputImage, imageWidth, imageHeight, imageChannels);
  


  wbCheck(cudaMemcpy(hostInputImageData, deviceInputImage, imageWidth * imageHeight * imageChannels * sizeof(float), cudaMemcpyDeviceToHost)); // copy the variables into gpu memory
  wbImage_setData(outputImage,hostInputImageData);

  wbSolution(args, outputImage);

  //@@ insert code here
  
  cudaFree(deviceInputImage);
  cudaFree(deviceColorImage);
  cudaFree(deviceGreyImage);
  cudaFree(deviceHistogram);
  cudaFree(deviceHistogramCdf);
  

  return 0;
}

