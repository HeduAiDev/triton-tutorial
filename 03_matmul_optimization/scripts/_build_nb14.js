const fs = require('fs');
const cells = [];
const BT = '`';
const BT3 = '```';

function md(source) {
  cells.push({
    cell_type: 'markdown', metadata: {},
    source: source.split('\n').map((l, i, a) => i < a.length - 1 ? l + '\n' : l)
  });
}

function code(source) {
  cells.push({
    cell_type: 'code', metadata: {},
    source: source.split('\n').map((l, i, a) => i < a.length - 1 ? l + '\n' : l),
    outputs: [], execution_count: null
  });
}

// ============================================================
// Cell 0: Chapter header
// ============================================================
md(`# 第14章：Split-K 并行 -- K维并行切分

## 前置知识
- 第09章：分块矩阵乘法基础
- 第12章：Block Pointer 与 Shared Memory
- 第13章：软件流水线

## 学习目标
- 理解标准 GEMM 在 **tall-skinny 矩阵** 上并行度不足的问题
- 掌握 **Split-K** 算法的原理
- 实现 Split-K GEMM kernel (在 Ch.13 pipeline kernel 基础上累进叠加)
- 了解 Split-K vs 标准 GEMM 的适用场景
- 与 **cuBLAS** 对比性能

## 累进优化
- ✅ Shared Memory / Block Pointer (Ch.12)
- ✅ 软件流水线 (Ch.13)
- ✅ **Split-K 并行 (本章新增)**

> **注意**: Split-K 是从 Ch.13 分出的一条**分支路径**, 适用于 tall-skinny 矩阵。
> Ch.15 的 Swizzle 是另一条分支路径, 适用于标准大方阵。
> Ch.18 的 ultimate kernel = smem + pipeline + swizzle (不含 Split-K)。`);

// Cell 1: imports
code(`import torch
import triton
import triton.language as tl`);

// Cell 2: 14.1 并行度不足问题
md(`## 14.1 标准 GEMM 的并行度问题

### 问题：tall-skinny 矩阵

${BT3}
Case 1: 方阵 (M=4096, N=4096, K=1024)
  BLOCK_M=128, BLOCK_N=128
  grid = (32, 32) = 1024 个 program
  → GPU (假设 80+ SMs) 可以完全利用 ✓

Case 2: tall-skinny (M=128, N=128, K=8192)
  grid = (1, 1) = 1 个 program!!
  → 只用了 1 个 SM, 其他 SM 空闲! ✗

Case 3: 中等宽矩阵 (M=256, N=256, K=16384)
  grid = (2, 2) = 4 个 program
  → 只用了 4 个 SM ✗
${BT3}

### Split-K 的解决方案

${BT3}
Split-K (SPLIT_K=8, M=128, N=128, K=8192):

K 维度被切成 8 份:
┌──────────┬──────────┬──────────┬──────────┬─...─┐
│ K chunk 0│ K chunk 1│ K chunk 2│ K chunk 3│     │
│ k=0:1024 │k=1024:2048│k=2048:3072│k=3072:4096│    │
└──────────┴──────────┴──────────┴──────────┴─...─┘
     ↓            ↓            ↓            ↓
  Program 0    Program 1    Program 2    Program 3  ...
     ↓            ↓            ↓            ↓
  C_partial_0  C_partial_1  C_partial_2  C_partial_3
     ↓            ↓            ↓            ↓
     └──────── atomic_add ────────────────┘
                     ↓
                C = Σ C_partial_i

并行度: 1 → 8 (提升 8x!)
${BT3}`);

// Cell 3: 14.2 实现 markdown
md(`## 14.2 实现: Split-K Pipeline GEMM

在 Ch.13 的 pipeline kernel 基础上累进叠加:
- grid 从 2D ${BT}(M_blocks, N_blocks)${BT} 改为 2D ${BT}(M_blocks * N_blocks, SPLIT_K)${BT}
- K 循环只处理 ${BT}[k_start, k_end)${BT} 范围
- 写回使用 ${BT}tl.atomic_add${BT} 合并部分和
- 保留 ${BT}num_stages${BT} 控制流水线深度`);

// Cell 4: Split-K kernel
code(`@triton.jit
def matmul_splitk_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    """
    Split-K Pipeline GEMM kernel.

    累进优化:
      ✅ make_block_ptr (Ch.12) — 结构化内存访问
      ✅ FP32 累加器 — 数值精度
      ✅ Tensor Core (tl.dot) — 自动映射
      ✅ num_stages (Ch.13) — 流水线 (通过 launch 参数)
      ✅ Split-K (本章) — K 维并行

    grid = (cdiv(M, BM) * cdiv(N, BN), SPLIT_K)
    """
    # ========== pid 解码 ==========
    pid_mn = tl.program_id(0)  # M-N 维度
    pid_k = tl.program_id(1)   # K 维度 split

    grid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid_mn // grid_n
    pid_n = pid_mn % grid_n

    # ========== 计算当前 split 的 K 范围 ==========
    k_per_split = tl.cdiv(K, SPLIT_K)
    k_start = pid_k * k_per_split
    k_end = min(k_start + k_per_split, K)

    # ========== Block Pointer (Ch.12) ==========
    a_block_ptr = tl.make_block_ptr(
        base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_M, k_start),
        block_shape=(BLOCK_M, BLOCK_K), order=(1, 0),
    )
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn),
        offsets=(k_start, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N), order=(1, 0),
    )

    # ========== FP32 累加器 ==========
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # ========== K 循环 (只处理分配的范围) + Pipeline (num_stages) ==========
    for k in range(k_start, k_end, BLOCK_K):
        a_tile = tl.load(a_block_ptr, boundary_check=(0, 1))
        b_tile = tl.load(b_block_ptr, boundary_check=(0, 1))
        acc = tl.dot(a_tile, b_tile, acc=acc)
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))

    # ========== 使用 atomic_add 写回部分和 ==========
    c = acc.to(tl.float16)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + rm[:, None] * stride_cm + rn[None, :] * stride_cn
    c_mask = (rm[:, None] < M) & (rn[None, :] < N)
    tl.atomic_add(c_ptrs, c, mask=c_mask)`);

// Cell 5: Host wrappers
code(`def matmul_splitk(a, b, BLOCK_M=128, BLOCK_N=128, BLOCK_K=32,
                   SPLIT_K=4, num_stages=3):
    """Split-K Pipeline GEMM。C 必须初始化为 0 (atomic_add 累加)。"""
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    c = torch.zeros((M, N), device=a.device, dtype=a.dtype)  # 必须为 0!
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N), SPLIT_K)
    matmul_splitk_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        SPLIT_K=SPLIT_K, num_stages=num_stages,
    )
    return c

# Ch.13 的标准 pipeline kernel (用于对比)
@triton.jit
def matmul_standard_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    a_block_ptr = tl.make_block_ptr(
        base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_K), order=(1, 0),
    )
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BLOCK_N), block_shape=(BLOCK_K, BLOCK_N), order=(1, 0),
    )
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a_tile = tl.load(a_block_ptr, boundary_check=(0, 1))
        b_tile = tl.load(b_block_ptr, boundary_check=(0, 1))
        acc = tl.dot(a_tile, b_tile, acc=acc)
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))
    c_block_ptr = tl.make_block_ptr(
        base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N), order=(1, 0),
    )
    tl.store(c_block_ptr, acc.to(tl.float16), boundary_check=(0, 1))

def matmul_standard(a, b, BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, num_stages=3):
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    matmul_standard_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_stages=num_stages,
    )
    return c`);

// Cell 6: correctness
code(`# ========== 正确性验证 ==========
torch.manual_seed(42)

print("Split-K 正确性验证:")
for M, N, K, SPLIT_K in [
    (128, 128, 2048, 4),
    (256, 256, 4096, 8),
    (512, 512, 8192, 4),
    (2048, 2048, 1024, 2),
]:
    a = torch.randn(M, K, device='cuda', dtype=torch.float16)
    b = torch.randn(K, N, device='cuda', dtype=torch.float16)
    c_splitk = matmul_splitk(a, b, SPLIT_K=SPLIT_K)
    c_ref = torch.matmul(a, b)
    max_err = (c_splitk - c_ref).abs().max().item()
    rel_err = torch.norm(c_splitk.float() - c_ref.float()) / torch.norm(c_ref.float())
    print(f"  ({M:>4}x{K:>5}) @ ({K:>5}x{N:>4}), SPLIT_K={SPLIT_K}: "
          f"max_err={max_err:.4f}, rel_err={rel_err:.6f}")`);

// Cell 7: 14.3 benchmark markdown
md(`## 14.3 性能对比: Standard vs Split-K vs cuBLAS

Split-K 在 tall-skinny 矩阵上应该显著优于标准 GEMM, 但在大方阵上因 atomic_add 开销可能更慢。`);

// Cell 8: benchmark utility
code(`# ========== benchmark 工具 ==========
def benchmark_fn(fn, a, b, num_warmup=25, num_rep=100, **kwargs):
    for _ in range(num_warmup):
        fn(a, b, **kwargs)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(num_rep):
        fn(a, b, **kwargs)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / num_rep

def benchmark_cublas(a, b, num_warmup=25, num_rep=100):
    for _ in range(num_warmup):
        torch.matmul(a, b)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(num_rep):
        torch.matmul(a, b)
    e.record()
    torch.cuda.synchronize()
    return s.elapsed_time(e) / num_rep`);

// Cell 9: Standard vs Split-K vs cuBLAS
code(`# ========== Standard vs Split-K vs cuBLAS ==========
BM, BN, BK = 128, 128, 32
gpu_name = torch.cuda.get_device_name(0)
num_sms = torch.cuda.get_device_properties(0).multi_processor_count
print(f"GPU: {gpu_name}, SMs: {num_sms}")

print(f"\\n{'Shape':>25} | {'#Progs':>7} {'Standard':>10} | {'SK':>3} {'#Progs':>7} {'Split-K':>10} | {'cuBLAS':>10} | {'SK加速':>7} {'SK/cuBLAS':>10}")
print("-" * 115)

test_cases = [
    (128,  128,  8192,  8),
    (256,  256,  8192,  8),
    (256,  256,  4096,  4),
    (512,  512,  4096,  4),
    (1024, 1024, 4096,  4),
    (2048, 2048, 1024,  2),
    (4096, 4096, 1024,  2),
]

for M, N, K, SPLIT_K in test_cases:
    a = torch.randn(M, K, device='cuda', dtype=torch.float16)
    b = torch.randn(K, N, device='cuda', dtype=torch.float16)

    progs_std = triton.cdiv(M, BM) * triton.cdiv(N, BN)
    progs_sk = progs_std * SPLIT_K

    ms_std = benchmark_fn(matmul_standard, a, b, num_stages=3)
    ms_sk = benchmark_fn(matmul_splitk, a, b, SPLIT_K=SPLIT_K, num_stages=3)
    ms_cu = benchmark_cublas(a, b)

    speedup = ms_std / ms_sk
    vs_cu = ms_cu / ms_sk
    shape_str = f"({M}x{K})@({K}x{N})"
    print(f"{shape_str:>25} | {progs_std:>7} {ms_std:>9.3f}ms | {SPLIT_K:>3} {progs_sk:>7} {ms_sk:>9.3f}ms | {ms_cu:>9.3f}ms | {speedup:>6.2f}x {vs_cu:>9.2f}x")`);

// Cell 10: 14.4 SPLIT_K tuning
md(`## 14.4 SPLIT_K 参数调优

对于给定的矩阵形状, 最优的 SPLIT_K 取决于标准并行度与 SM 数量的关系:

${BT3}
决策规则:
  standard_programs = cdiv(M, BM) * cdiv(N, BN)
  if standard_programs < num_SMs * 2:
    → GPU 利用率不足, 使用 Split-K
    → SPLIT_K ≈ ceil(num_SMs * 2 / standard_programs)
  else:
    → 标准 GEMM 已有足够并行度, atomic_add 开销不值得
${BT3}`);

// Cell 11: SPLIT_K sweep
code(`# ========== 不同 SPLIT_K 值的效果 ==========
M, N, K = 256, 256, 8192
a = torch.randn(M, K, device='cuda', dtype=torch.float16)
b = torch.randn(K, N, device='cuda', dtype=torch.float16)

print(f"矩阵: ({M}x{K}) @ ({K}x{N})")
print(f"标准 programs: {triton.cdiv(M,128) * triton.cdiv(N,128)}")
print(f"GPU SMs: {num_sms}")
flops = 2.0 * M * N * K

ms_cu = benchmark_cublas(a, b)
tf_cu = flops / (ms_cu * 1e-3) / 1e12
print(f"cuBLAS: {ms_cu:.3f}ms, {tf_cu:.1f} TFLOPS")

print(f"\\n{'Method':>10} | {'Programs':>10} | {'时间(ms)':>10} | {'TFLOPS':>8} | {'vs cuBLAS':>10}")
print("-" * 60)

# Standard
ms_std = benchmark_fn(matmul_standard, a, b, num_stages=3)
tf_std = flops / (ms_std * 1e-3) / 1e12
print(f"{'标准':>10} | {triton.cdiv(M,128)*triton.cdiv(N,128):>10} | {ms_std:>10.3f} | {tf_std:>8.1f} | {ms_cu/ms_std:>9.2f}x")

# Split-K sweep
for sk in [2, 4, 8, 16, 32]:
    progs = triton.cdiv(M, 128) * triton.cdiv(N, 128) * sk
    ms = benchmark_fn(matmul_splitk, a, b, SPLIT_K=sk, num_stages=3)
    tf = flops / (ms * 1e-3) / 1e12
    print(f"{'SK='+str(sk):>10} | {progs:>10} | {ms:>10.3f} | {tf:>8.1f} | {ms_cu/ms:>9.2f}x")`);

// Cell 12: 14.5 总结
md(`## 14.5 总结

### 本章要点

1. **并行度不足问题**: 当 M,N 较小时, 标准 GEMM 的 grid 太小, 无法充分利用 GPU

2. **Split-K 算法**:
   - 将 K 维度切成 SPLIT_K 份
   - 每份由不同的 program 计算部分和
   - 使用 ${BT}tl.atomic_add${BT} 合并结果
   - 并行度从 ${BT}cdiv(M,BM)*cdiv(N,BN)${BT} 提升到 ${BT}cdiv(M,BM)*cdiv(N,BN)*SPLIT_K${BT}

3. **累进叠加**: 在 Ch.13 的 pipeline kernel 基础上添加 Split-K, 同时保留 ${BT}num_stages${BT}

4. **适用场景**:
   - ✅ M,N 小但 K 大的 tall-skinny 矩阵
   - ✗ 大方阵 (标准 GEMM 已有足够并行度, atomic_add 反而拖慢)

5. **决策规则**: ${BT}standard_programs < num_SMs * 2${BT} 时考虑 Split-K

### 累进优化状态
| 特性 | 状态 | 路径 |
|------|------|------|
| Shared Memory / Block Pointer | ✅ Ch.12 | 主线 |
| 软件流水线 (num_stages) | ✅ Ch.13 | 主线 |
| **Split-K 并行** | **✅ 本章** | **分支** (tall-skinny) |
| L2 Cache Swizzle | → Ch.15 | 分支 (标准矩阵) |

### 下一章预告

第15章将介绍 **L2 Cache Swizzle**, 通过重排 block 调度顺序来优化 L2 缓存命中率。
Swizzle 是从 Ch.13 分出的另一条分支路径, 适用于标准大方阵, 将在 Ch.18 ultimate 中使用。`);

// Write
const nb = {
  nbformat: 4, nbformat_minor: 5,
  metadata: {
    kernelspec: { display_name: 'Python 3', language: 'python', name: 'python3' },
    language_info: { name: 'python', version: '3.10.0' }
  },
  cells
};

const outPath = '../14_matmul_splitk.ipynb';
fs.writeFileSync(outPath, JSON.stringify(nb, null, 1));
console.log(`Wrote ${outPath} (${cells.length} cells, ${fs.statSync(outPath).size} bytes)`);
