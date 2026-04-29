/*
 * SIMT Naive GEMM Kernel — extracted from src/simt/00simt_naive.cu
 * ================================================================
 * One thread per output element — the simplest possible GPU GEMM.
 *
 * Block tile: BlockTileM=16 × BlockTileN=8
 * Accumulation: FP16 (lowest precision baseline, matches original)
 *
 * Original performance: 6.7966 ms, M=N=2048, K=1024, RTX PRO 6000 Blackwell
 *
 * This file is compiled as a PyTorch CUDA extension so it can be called
 * from Python benchmarks via benchmark_utils.benchmark_method().
 */

#include <torch/extension.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

// ── Kernel (faithful reproduction of 00simt_naive.cu) ─────────────

__global__ void simt_naive_kernel(
    const __half* __restrict__ A,
    const __half* __restrict__ B,
    __half* __restrict__ C,
    int M, int N, int K)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (col >= N || row >= M) return;

    // FP16 accumulation — matches original "half sum = 0;"
    __half sum = __float2half(0.0f);
    for (int k = 0; k < K; k++) {
        __half a_val = A[row * K + k];
        __half b_val = B[k * N + col];
        sum = __hadd(sum, __hmul(a_val, b_val));
    }
    C[row * N + col] = sum;
}

// ── PyTorch host wrapper ─────────────────────────────────────────

torch::Tensor simt_naive_cuda(const torch::Tensor& A, const torch::Tensor& B) {
    TORCH_CHECK(A.dim() == 2 && B.dim() == 2,
                "Expected 2D tensors");
    TORCH_CHECK(A.size(1) == B.size(0),
                "K dimension mismatch");
    TORCH_CHECK(A.scalar_type() == torch::kFloat16 &&
                B.scalar_type() == torch::kFloat16,
                "Inputs must be float16");
    TORCH_CHECK(A.is_cuda(), "Tensors must be on CUDA");

    int M = A.size(0), K_ = A.size(1), N = B.size(1);
    auto C = torch::empty({M, N}, A.options());

    const int BM = 16, BN = 8;
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    dim3 block(BN, BM);

    simt_naive_kernel<<<grid, block>>>(
        reinterpret_cast<const __half*>(A.data_ptr<at::Half>()),
        reinterpret_cast<const __half*>(B.data_ptr<at::Half>()),
        reinterpret_cast<__half*>(C.data_ptr<at::Half>()),
        M, N, K_
    );
    return C;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("simt_naive_cuda", &simt_naive_cuda,
          "SIMT Naive FP16 GEMM — one thread per element (00simt_naive.cu)");
}
