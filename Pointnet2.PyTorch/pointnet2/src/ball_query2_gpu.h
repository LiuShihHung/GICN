#ifndef _BALL_QUERY2_GPU_H
#define _BALL_QUERY2_GPU_H

#include <torch/serialize/tensor.h>
#include <vector>
#include <cuda.h>
#include <cuda_runtime_api.h>

int ball_query2_wrapper_fast(int b, int n, int m, int nsample, at::Tensor radius_size_tensor,
	at::Tensor new_xyz_tensor, at::Tensor xyz_tensor, at::Tensor idx_tensor);

void ball_query2_kernel_launcher_fast(int b, int n, int m,  int nsample, const float *radius_size,
	const float *xyz, const float *new_xyz, int *idx, cudaStream_t stream);

#endif
