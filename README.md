# Triton 从入门到专家：GEMM终极优化 & FlashAttention 全系列

> 基于 CUDA HGEMM 优化项目，用 Triton 重新实现矩阵乘法的完整优化路径，并实现 FlashAttention v1-v4。

## 环境要求

- Python 3.10+
- PyTorch 2.0+
- Triton 2.2+
- NVIDIA GPU (Compute Capability >= 7.0)

```bash
pip install -r requirements.txt
```

## 教程结构

### Part 1: Triton 基础 (`01_basics/`)

| 序号 | 主题 | 内容 |
|------|------|------|
| 01 | Hello Triton | 第一个 kernel：向量加法，理解 Triton 编程模型 |
| 02 | Program ID | 执行模型、grid、多维索引 |
| 03 | 内存操作 | 指针运算、load/store、mask、边界处理 |
| 04 | Block 操作 | tl.arange、广播、2D 索引模式 |
| 05 | 控制流与归约 | tl.where、tl.sum/max/min、原子操作 |
| 06 | 数学运算与类型 | exp/log/sqrt、类型转换、constexpr |

### Part 2: 核心模式 (`02_intermediate/`)

| 序号 | 主题 | 内容 |
|------|------|------|
| 07 | Softmax | 在线 softmax、行级归约、数值稳定性 |
| 08 | 矩阵乘法（朴素） | 对应 CUDA `00simt_naive`，理解性能瓶颈 |
| 09 | 矩阵乘法（分块） | 对应 CUDA `01simt_regci`，K 维分块 + tl.dot |
| 10 | 算子融合 | MatMul + Bias + ReLU 融合、LayerNorm |
| 11 | 自动调优 | @triton.autotune、Config、性能基准测试 |

### Part 3: GEMM 终极优化 (`03_matmul_optimization/`)

| 序号 | 主题 | 对应 CUDA 实现 |
|------|------|---------------|
| 12 | 共享内存控制 | `02simt_smem` → block_ptr |
| 13 | 软件流水线 | `04simt_pipline` → num_stages (smem+pipeline) |
| 14 | Split-K 并行 | K 维并行切分 (smem+pipeline+splitk) |
| 15 | L2 Cache Swizzle | `03simt_smemT` → L2 缓存优化 (smem+pipeline+swizzle) |
| 16 | 混合精度策略 | FP16/FP32/TF32/BF16 累加 |
| 17 | Tensor Core 深入 | 对应 `wmma/mma` → tl.dot 映射 |
| 18 | 终极优化 GEMM | 所有技巧组合 + 对比 cuBLAS |

### Part 4: FlashAttention 全系列 (`04_flash_attention/`)

| 序号 | 主题 | 内容 |
|------|------|------|
| 19 | 标准 Attention | 朴素实现，O(N²) 内存分析 |
| 20 | FlashAttention v1 | 分块 + 在线 softmax，O(N) 内存 |
| 21 | FlashAttention v2 | Q 外循环、减少非 matmul FLOPs |
| 22 | FlashAttention v3 | 异步数据搬运、warp 特化概念 |
| 23 | FlashAttention v4 | 块稀疏、变长序列、GQA 支持 |

### Part 5: 高级专题 (`05_advanced/`)

| 序号 | 主题 | 内容 |
|------|------|------|
| 24 | 调试与性能分析 | TRITON_INTERPRET、device_print、NSight |
| 25 | 编译器 IR | TTIR → TTGIR → LLVM IR → PTX |
| 26 | 生产集成 | autograd.Function、torch.compile |
| 27 | PyPTO vs Triton | 跨平台矩阵乘法对比：昇腾 NPU 编程 |

## CUDA → Triton 概念映射

| CUDA 概念 | Triton 对应 |
|-----------|------------|
| `threadIdx.x` | 隐式（块级编程） |
| `blockIdx.x` | `tl.program_id(0)` |
| `__shared__` | 自动管理 |
| `__syncthreads()` | 隐式同步 |
| `float4` 向量化加载 | 自动合并访问 |
| `wmma::mma_sync` | `tl.dot()` |
| `ldmatrix` PTX | 编译器自动生成 |
| 软件流水线 | `num_stages` 参数 |
| Bank Conflict 避免 | 自动 Swizzle |

## 性能目标

```
朴素 GEMM:           ~14.8 ms  (baseline)
分块 GEMM:           ~1.2 ms   (12x↑)
共享内存 GEMM (Ch12): ~1.1 ms   (smem)
流水线 GEMM (Ch13):   ~0.7 ms   (smem+pipeline)
Swizzle GEMM (Ch15):  ~0.6 ms   (smem+pipeline+swizzle)
Tensor Core (Ch17):   ~0.3 ms   (all optimizations)
cuBLAS:              ~0.25 ms  (参考)
```

*测试条件：M=N=2048, K=1024, FP16*
