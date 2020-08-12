#include <torch/serialize/tensor.h>
#include <vector>
#include <THC/THC.h>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "ball_query2_gpu.h"

extern THCState *state;

#define CHECK_CUDA(x) TORCH_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x, " must be contiguous ")
#define CHECK_INPUT(x) CHECK_CUDA(x);CHECK_CONTIGUOUS(x)

int ball_query2_wrapper_fast(int b, int n, int m, int nsample, at::Tensor radius_size_tensor,
    at::Tensor new_xyz_tensor, at::Tensor xyz_tensor, at::Tensor idx_tensor) {
    CHECK_INPUT(radius_size_tensor);
    CHECK_INPUT(new_xyz_tensor);
    CHECK_INPUT(xyz_tensor);
    const float *radius_size = radius_size_tensor.data<float>();
    const float *new_xyz = new_xyz_tensor.data<float>();
    const float *xyz = xyz_tensor.data<float>();
    int *idx = idx_tensor.data<int>();
    
    cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    ball_query2_kernel_launcher_fast(b, n, m, nsample, radius_size ,new_xyz, xyz, idx, stream);
    return 1;
}