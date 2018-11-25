# Applied-Parallel-Programming-Project
Purpose of the project:
Get practical experience by using, profiling, and modifying MXNet, a standard open-source neural-network framework.
Demonstrate command of CUDA and optimization approaches by designing and implementing an optimized neural-network convolution layer forward pass.

# Mp1 
Implementation of a simple vector addition kernel and its associated host code

# Mp2
Implementation of a basic dense matrix multiplication routine.

# Mp3
 Implementation of a tiled dense matrix multiplication routine using shared memory
 
 # Mp4
 Implementation of a 3D convolution using constant memory for the kernel and 3D shared memory tiling
 
 # Mp5.1
Implement a kernel and associated host code that performs reduction of a 1D list stored in a C array. The reduction should give the sum of the list. You should implement the improved kernel discussed in the lecture. Your kernel should be able to handle input lists of arbitrary length.

# Mp5.2
Implement one or more kernels and their associated host code to perform parallel scan on a 1D list. The scan operator used will be addition. You should implement the work- efficient kernel discussed in lecture. Your kernel should be able to handle input lists of arbitrary length. However, for simplicity, you can assume that the input list will be at most 2,048 * 2,048 elements.

# Mp6
Implement an efficient histogramming equalization algorithm for an input image. Like the image convolution MP, the image is represented as RGB float values. You will convert that to GrayScale unsigned char values and compute the histogram. Based on the histogram, you will compute a histogram equalization function which you will then apply to the original image to get the color corrected image.
 
# Mp7
Implement a SpMV (Sparse Matrix Vector Multiplication) kernel for an input sparse matrix based on the Jagged Diagonal Storage (JDS) transposed format.

# General Mp Instructions
Allocate device memory
Copy host memory to device
Initialize thread block and kernel grid dimensions
Invoke CUDA kernel
Copy results from device to host
Deallocate device memory
Implement the matrix-matrix multiplication routine using shared memory and tiling
