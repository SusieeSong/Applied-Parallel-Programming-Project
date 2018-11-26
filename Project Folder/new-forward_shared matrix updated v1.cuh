
#ifndef MXNET_OPERATOR_NEW_FORWARD_CUH_
#define MXNET_OPERATOR_NEW_FORWARD_CUH_

#define CUDA_MAX_NUM_THREADS 1024

#define TILE_WIDTH 16





#include <mxnet/base.h>

namespace mxnet
{
namespace op
{



__global__ void X_unroll(const int C, const int H, const int W, const int K, float * x, float * x_unroll){
    

    int b = blockIdx.x;
    int t = blockIdx.y * CUDA_MAX_NUM_THREADS + threadIdx.x; 

    int c, s, h_out, w_out, h_unroll, w_unroll, h_base, p, q;

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    const int H_unroll = C * K * K;
    const int W_unroll = H_out * W_out;

    #define x_unroll(i2, i1, i0) x_unroll[ (i2) * W_unroll * H_unroll + (i1) * W_unroll + i0 ]
    #define x4d(i3, i2, i1, i0) x[(i3)*(C * H * W) + (i2)*(H * W) + (i1)*(W) + i0]

    if (t < C * W_unroll){
        c = t / W_unroll;
        s = t % W_unroll; 
        h_out = s / W_out;
        w_out = s % W_out;
        w_unroll = h_out * W_out + w_out;
        h_base = c * K * K;
        for(p = 0; p < K; p++)
            for(q = 0; q < K; q++){
                h_unroll = h_base + p * K + q;
                x_unroll(b, h_unroll, w_unroll) = x4d(b, c, h_out + p, w_out + q);
            }
    }

    #undef x_unroll
    #undef x4d
}




__global__ void MatrixMultiplyKernel(const int H_unroll, const int M, const int W_unroll, float * x_unroll, float * w, float * y){
  
    __shared__ float w_tile[TILE_WIDTH][TILE_WIDTH * 2], x_unroll_tile[TILE_WIDTH * 2][TILE_WIDTH];

    const int b = blockIdx.x;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int Row = blockIdx.z * TILE_WIDTH + ty;
    int Col = blockIdx.y * TILE_WIDTH + tx;

    float CValue = 0;
    int i, ph;

    #define x3d_unroll(i2, i1, i0) x_unroll[(i2)*(H_unroll * W_unroll) + (i1)*(W_unroll) + i0 ]
    #define w2d(i1,i0) w[(i1)*H_unroll + i0]
    #define y3d(i2, i1, i0) y[(i2)*(M * W_unroll) + (i1)*(W_unroll) + i0]
 
    if(Row < M && tx < H_unroll && tx < TILE_WIDTH)
            w_tile[ty][tx] = w2d(Row, tx);
        
    if(ty < H_unroll && Col < W_unroll)  
            x_unroll_tile[ty][tx] = x3d_unroll(b, ty, Col); 
     __syncthreads();

    for(ph = 0; ph < ((H_unroll - TILE_WIDTH - 1) / (2 * TILE_WIDTH) + 1); ph++){
        if(Row < M && (ph * 2 * TILE_WIDTH + tx + TILE_WIDTH) < H_unroll)
            w_tile[ty][tx + TILE_WIDTH] = w2d(Row, ph * 2 * TILE_WIDTH + tx + TILE_WIDTH);
        
        if((ph * TILE_WIDTH * 2 + ty + TILE_WIDTH) < H_unroll && Col < W_unroll)
            x_unroll_tile[ty + TILE_WIDTH][tx] = x3d_unroll(b, ph * 2 * TILE_WIDTH + ty + TILE_WIDTH, Col);
        
        for(i = 0; i < TILE_WIDTH; ++i){
                CValue += w_tile[ty][i] * x_unroll_tile[i][tx];   
        }

        __syncthreads();

        if(Row < M && (ph * 2 * TILE_WIDTH + tx + 2 * TILE_WIDTH) < H_unroll)
            w_tile[ty][tx] = w2d(Row, ph * 2 * TILE_WIDTH + tx + 2 * TILE_WIDTH);
        
        if((ph * TILE_WIDTH * 2 + ty + 2 * TILE_WIDTH) < H_unroll && Col < W_unroll)
            x_unroll_tile[ty][tx] = x3d_unroll(b, ph * 2 * TILE_WIDTH + ty + 2 * TILE_WIDTH, Col);


        for(i = TILE_WIDTH; i < 2 * TILE_WIDTH; ++i){
            
                CValue += w_tile[ty][i] * x_unroll_tile[i][tx];   
        }
        __syncthreads();
    }

    if(Row < M && Col < W_unroll)
        y3d(b, Row, Col) = CValue;

    #undef x3d_unroll
    #undef w2d
    #undef y3d
}

/* 
   This function is called by new-inl.h
   Any code you write should be executed by this function.
   For ECE408, we only expect the float version of the operator to be called, so here we specialize with only floats.
*/
template<>
void forward<gpu, float>(mshadow::Tensor<gpu, 4, float> &y, const mshadow::Tensor<gpu, 4, float> &x, const mshadow::Tensor<gpu, 4, float> &w) {
    

    // Use mxnet's CHECK_EQ to do assertions.
    // Remove this assertion when you do your implementation!
    //CHECK_EQ(0, 1) << "Missing an ECE408 GPU implementation!";

    // You'll probably need to launch kernels against the right stream to keep MXNet happy
    cudaStream_t s = y.stream_->stream_;
    

    // Extract the tensor dimensions into B,M,C,H,W,K
    const int B = x.shape_[0];
    const int M = y.shape_[1];
    const int C = x.shape_[1];
    const int H = x.shape_[2];
    const int W = x.shape_[3];
    const int K = w.shape_[3];

    const int H_out = H - K + 1;
    const int W_out = W - K + 1;

    const int H_unroll = C * K * K;
    const int W_unroll = H_out * W_out;

    float *x_unroll;
    cudaMalloc((void **)&x_unroll, sizeof(float) * H_unroll * W_unroll*B);


    const int threads_NUM = C * H_out * W_out;
    const int blocks_NUM = ceil(threads_NUM / (float)CUDA_MAX_NUM_THREADS);
    
    dim3 unroll_gridDim(B, blocks_NUM, 1);
    dim3 unroll_blockDim(CUDA_MAX_NUM_THREADS, 1, 1);

    X_unroll<<<unroll_gridDim, unroll_blockDim, 0, s>>>(C, H, W, K, x.dptr_, x_unroll);

    dim3 Matrix_gridDim(B, ceil(W_unroll/(float)TILE_WIDTH), ceil(M/(float)TILE_WIDTH));
    dim3 Matrix_blockDim(TILE_WIDTH, TILE_WIDTH, 1);
    
   
    MatrixMultiplyKernel<<<Matrix_gridDim, Matrix_blockDim, 0, s>>>(H_unroll, M, W_unroll, x_unroll, w.dptr_, y.dptr_);


    MSHADOW_CUDA_CALL(cudaDeviceSynchronize());

    cudaFree(x_unroll);

}


/* 
    This tells mxnet how to do an op when it's not a float.
    This is not used in the ECE408 project
*/
template<typename gpu, typename DType>
void forward(mshadow::Tensor<gpu, 4, DType> &y, const mshadow::Tensor<gpu, 4, DType> &x, const mshadow::Tensor<gpu, 4, DType> &w) {
    assert( 0 && "No forward implementation for other datatypes needed for ECE408");
}

}
}

#endif
