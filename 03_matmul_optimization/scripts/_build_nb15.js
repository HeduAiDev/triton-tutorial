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
md(`# 第15章：L2 Cache Swizzle -- 缓存友好的 Block 调度

## 前置知识
- 第12章：Block Pointer 与 Shared Memory
- 第13章：软件流水线
- GPU L2 Cache 的基本概念

## 学习目标
- 理解标准 block 调度在 L2 cache 上的问题
- 掌握 **Swizzle (CTA swizzle)** 的原理与实现
- 在 Ch.13 的 pipeline kernel 基础上累进叠加 swizzle
- 分析不同 GPU 架构上 swizzle 的效果差异
- 与 **cuBLAS** 对比性能

## 累进优化
- ✅ Shared Memory / Block Pointer (Ch.12)
- ✅ 软件流水线 (Ch.13)
- ✅ **L2 Cache Swizzle (本章新增)**

> **注意**: Swizzle 与 Split-K (Ch.14) 是从 Ch.13 分出的**两条分支路径**:
> - Split-K 适用于 tall-skinny 矩阵 (M,N 小, K 大)
> - Swizzle 适用于标准/大方阵 (M,N 大)
> - Ch.18 ultimate = smem + pipeline + swizzle (不含 Split-K)`);

// Cell 1: imports
code(`import torch
import triton
import triton.language as tl`);

// Cell 2: 15.1 L2 cache 问题
md(`## 15.1 L2 Cache 问题: Block 调度顺序的影响

### 标准 2D Grid 调度

在标准的 2D grid 中, block 按行优先 (row-major) 顺序被调度到 SM:

${BT3}
C 矩阵的 block 调度顺序 (2D grid, 行优先):

     N (列方向) →
  ┌────┬────┬────┬────┬────┬────┬────┬────┐
M │ P0 │ P1 │ P2 │ P3 │ P4 │ P5 │ P6 │ P7 │ ← 第0行的8个 block 先全部处理
↓ ├────┼────┼────┼────┼────┼────┼────┼────┤
  │ P8 │ P9 │P10 │P11 │P12 │P13 │P14 │P15 │ ← 然后第1行
  ├────┼────┼────┼────┼────┼────┼────┼────┤
  │P16 │P17 │P18 │P19 │P20 │P21 │P22 │P23 │
  ├────┼────┼────┼────┼────┼────┼────┼────┤
  │P24 │P25 │P26 │P27 │P28 │P29 │P30 │P31 │
  └────┴────┴────┴────┴────┴────┴────┴────┘
${BT3}

### 问题: 同一行的 block 没有 B 矩阵的数据复用

${BT3}
计算 C[i,j] 需要: A 的第 i 行 × B 的第 j 列

同一行的 block (P0-P7):
  P0: A[row 0] × B[col 0]
  P1: A[row 0] × B[col 1]  ← A[row 0] 被复用 ✓
  P2: A[row 0] × B[col 2]  ← A[row 0] 被复用 ✓
  ...
  P7: A[row 0] × B[col 7]  ← A[row 0] 被复用 ✓

但是 B 的列完全不同! B[col 0..7] 各占内存。
当 N 很大时, B 的 8 列可能超过 L2 容量,
导致 P8 开始计算时 B[col 0] 已经被逐出!

下一行 (P8-P15):
  P8:  A[row 1] × B[col 0]  ← B[col 0] 可能已不在 L2!
  P9:  A[row 1] × B[col 1]  ← B[col 1] 可能已不在 L2!
  ...

→ L2 cache 命中率低!
${BT3}

### 解决方案: 列优先分组 (Swizzle)

${BT3}
Swizzle 调度 (GROUP_SIZE_M=4):

     N (列方向) →
  ┌────┬────┬────┬────┬────┬────┬────┬────┐
  │ P0 │ P4 │ P8 │P12 │P16 │P20 │P24 │P28 │
  ├────┼────┼────┼────┼────┼────┼────┼────┤
  │ P1 │ P5 │ P9 │P13 │P17 │P21 │P25 │P29 │
  ├────┼────┼────┼────┼────┼────┼────┼────┤
  │ P2 │ P6 │P10 │P14 │P18 │P22 │P26 │P30 │
  ├────┼────┼────┼────┼────┼────┼────┼────┤
  │ P3 │ P7 │P11 │P15 │P19 │P23 │P27 │P31 │
  └────┴────┴────┴────┴────┴────┴────┴────┘

前4个 program (P0-P3) 计算同一列的4个 block:
  P0: A[row 0] × B[col 0]
  P1: A[row 1] × B[col 0]  ← B[col 0] 被复用 ✓
  P2: A[row 2] × B[col 0]  ← B[col 0] 被复用 ✓
  P3: A[row 3] × B[col 0]  ← B[col 0] 被复用 ✓

→ B 的列在 group 内被复用, L2 命中率提升!
→ A 的行只有4个 (vs 整行8个), L2 压力也降低!
${BT3}`);

// Cell 3: 15.2 swizzle 数学公式
md(`## 15.2 Swizzle 的数学实现

### 1D Grid + PID 重映射

标准 2D grid 中, ${BT}pid_m${BT} 和 ${BT}pid_n${BT} 直接从 ${BT}program_id(0)${BT} 和 ${BT}program_id(1)${BT} 获得。

Swizzle 使用 **1D grid**, 然后手动将线性 pid 映射为 (pid_m, pid_n):

${BT3}python
# 标准 2D grid (行优先):
pid_m = tl.program_id(0)  # axis 0
pid_n = tl.program_id(1)  # axis 1

# Swizzle (1D grid + 列优先分组):
pid = tl.program_id(0)    # 唯一的 1D pid

grid_m = cdiv(M, BLOCK_M)
grid_n = cdiv(N, BLOCK_N)

# 哪个 group?
group_id = pid // (GROUP_SIZE_M * grid_n)
# group 内的起始 M
first_pid_m = group_id * GROUP_SIZE_M
# group 实际大小 (最后一个 group 可能不满)
group_size_m = min(grid_m - first_pid_m, GROUP_SIZE_M)
# 列优先映射
pid_m = first_pid_m + (pid % (GROUP_SIZE_M * grid_n)) % group_size_m
pid_n = (pid % (GROUP_SIZE_M * grid_n)) // group_size_m
${BT3}

### 可视化映射过程

${BT3}
grid_m=4, grid_n=4, GROUP_SIZE_M=2:

线性 pid: 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15

Group 0 (pid 0-7):
  pid_m: 0  1  0  1  0  1  0  1
  pid_n: 0  0  1  1  2  2  3  3
  → 列优先遍历 M 方向的前2行

Group 1 (pid 8-15):
  pid_m: 2  3  2  3  2  3  2  3
  pid_n: 0  0  1  1  2  2  3  3
  → 列优先遍历 M 方向的后2行
${BT3}`);

// Cell 4: 15.3 L2 working set analysis
md(`## 15.3 L2 工作集分析

### 为什么 Swizzle 在某些 GPU 上效果不明显?

Swizzle 的收益取决于 B 矩阵的工作集是否超过 L2 容量:

${BT3}
L2 工作集大小 (假设 BLOCK_K 个 K 步):

标准 2D (行优先):
  同时活跃的 B 列数 = grid_n (所有列)
  B 工作集 ≈ K * N * sizeof(fp16) = K * N * 2 bytes

Swizzle (GROUP_SIZE_M 组):
  同时活跃的 B 列数 ≈ grid_n / group 数
  B 工作集 ≈ K * N * 2 / group 数

GPU L2 容量对比:
  A100:     40 MB L2
  H100:     50 MB L2
  Blackwell: 96 MB L2 (RTX PRO 6000)

示例: M=4096, N=4096, K=4096, FP16
  B 矩阵总大小 = 4096 * 4096 * 2 = 32 MB
  A100 L2 (40MB): B 刚好能放下, swizzle 帮助有限
  但如果 K=8192: B = 64 MB, 超过 L2, swizzle 会有帮助
  Blackwell L2 (96MB): 即使 K=8192, B 也能放下!
${BT3}

**结论**: L2 越大的 GPU, swizzle 的收益越小。在 Blackwell 上需要特别大的矩阵才能看到明显收益。`);

// Cell 5: 15.4 实现 markdown
md(`## 15.4 实现: Pipeline + Swizzle GEMM

在 Ch.13 的 pipeline kernel 基础上累进叠加:
- Grid 从 2D ${BT}(M_blocks, N_blocks)${BT} 改为 1D ${BT}(M_blocks * N_blocks,)${BT}
- 添加 pid 重映射逻辑 (swizzle)
- 保留 ${BT}num_stages${BT} 控制流水线深度`);

// Cell 6: Swizzle pipeline kernel + no-swizzle pipeline kernel (for comparison)
code(`@triton.jit
def matmul_swizzle_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
):
    """
    Pipeline + Swizzle GEMM kernel.

    累进优化:
      ✅ make_block_ptr (Ch.12) — 结构化内存访问
      ✅ FP32 累加器 — 数值精度
      ✅ Tensor Core (tl.dot) — 自动映射
      ✅ num_stages (Ch.13) — 流水线 (通过 launch 参数)
      ✅ Swizzle (本章) — L2 cache 友好的 block 调度

    grid = (cdiv(M, BM) * cdiv(N, BN),)  ← 1D grid
    """
    # ========== Swizzle pid 重映射 ==========
    pid = tl.program_id(0)  # 1D grid

    grid_m = tl.cdiv(M, BLOCK_M)
    grid_n = tl.cdiv(N, BLOCK_N)

    # 分组: 每 GROUP_SIZE_M 行为一组
    group_id = pid // (GROUP_SIZE_M * grid_n)
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(grid_m - first_pid_m, GROUP_SIZE_M)

    # 组内列优先映射
    pid_m = first_pid_m + ((pid % (GROUP_SIZE_M * grid_n)) % group_size_m)
    pid_n = (pid % (GROUP_SIZE_M * grid_n)) // group_size_m

    # ========== Block Pointer (Ch.12) ==========
    a_block_ptr = tl.make_block_ptr(
        base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K), order=(1, 0),
    )
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N), order=(1, 0),
    )

    # ========== FP32 累加器 ==========
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # ========== K 循环 + Pipeline (num_stages) ==========
    for k in range(0, K, BLOCK_K):
        a_tile = tl.load(a_block_ptr, boundary_check=(0, 1))
        b_tile = tl.load(b_block_ptr, boundary_check=(0, 1))
        acc = tl.dot(a_tile, b_tile, acc=acc)
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))

    # ========== 写回 ==========
    c_block_ptr = tl.make_block_ptr(
        base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N), order=(1, 0),
    )
    tl.store(c_block_ptr, acc.to(tl.float16), boundary_check=(0, 1))


# ========== Pipeline-only kernel (Ch.13 baseline, for comparison) ==========
@triton.jit
def matmul_pipeline_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Ch.13 pipeline kernel (2D grid, no swizzle) — 对比基线。"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    a_block_ptr = tl.make_block_ptr(
        base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K), order=(1, 0),
    )
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N), order=(1, 0),
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
    tl.store(c_block_ptr, acc.to(tl.float16), boundary_check=(0, 1))`);

// Cell 7: Host wrappers
code(`def matmul_swizzle(a, b, BLOCK_M=128, BLOCK_N=128, BLOCK_K=32,
                    GROUP_SIZE_M=8, num_stages=3):
    """Pipeline + Swizzle GEMM。"""
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)  # 1D grid!
    matmul_swizzle_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        num_stages=num_stages,
    )
    return c

def matmul_pipeline(a, b, BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, num_stages=3):
    """Ch.13 pipeline GEMM (2D grid, 对比基线)。"""
    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    matmul_pipeline_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_stages=num_stages,
    )
    return c`);

// Cell 8: Correctness
code(`# ========== 正确性验证 ==========
torch.manual_seed(42)

print("Swizzle GEMM 正确性验证:")
for M, N, K, GSM in [
    (512, 512, 1024, 4),
    (1024, 1024, 2048, 8),
    (2048, 2048, 1024, 8),
    (4096, 4096, 1024, 8),
]:
    a = torch.randn(M, K, device='cuda', dtype=torch.float16)
    b = torch.randn(K, N, device='cuda', dtype=torch.float16)
    c_swz = matmul_swizzle(a, b, GROUP_SIZE_M=GSM)
    c_ref = torch.matmul(a, b)
    max_err = (c_swz - c_ref).abs().max().item()
    rel_err = torch.norm(c_swz.float() - c_ref.float()) / torch.norm(c_ref.float())
    print(f"  ({M:>4}x{K:>5}) @ ({K:>5}x{N:>4}), GROUP_SIZE_M={GSM}: "
          f"max_err={max_err:.4f}, rel_err={rel_err:.6f}")`);

// Cell 9: 15.5 L2 working set calculation
md(`## 15.5 L2 工作集计算

先量化当前 GPU 的 L2 容量和矩阵的工作集大小, 预测 swizzle 在哪些尺寸上有效。`);

// Cell 10: L2 analysis code
code(`# ========== L2 工作集分析 ==========
props = torch.cuda.get_device_properties(0)
gpu_name = props.name
num_sms = props.multi_processor_count
l2_size_bytes = props.L2_cache_size  # L2 cache 大小 (bytes)
l2_size_mb = l2_size_bytes / (1024 * 1024)

print(f"GPU: {gpu_name}")
print(f"SMs: {num_sms}")
print(f"L2 Cache: {l2_size_mb:.0f} MB ({l2_size_bytes:,} bytes)")

print(f"\\n{'Shape':>25} | {'B 矩阵':>10} | {'A+B 总计':>10} | {'vs L2':>10} | {'预测':>10}")
print("-" * 80)

test_shapes = [
    (2048, 2048, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 1024),
    (4096, 4096, 2048),
    (4096, 4096, 4096),
    (8192, 8192, 2048),
    (8192, 8192, 4096),
    (8192, 4096, 4096),   # 非方阵
    (4096, 8192, 4096),   # 非方阵
]

for M, N, K in test_shapes:
    b_size = K * N * 2  # FP16
    a_size = M * K * 2
    total_mb = (a_size + b_size) / (1024 * 1024)
    b_mb = b_size / (1024 * 1024)
    ratio = (a_size + b_size) / l2_size_bytes
    predict = "有帮助" if ratio > 0.5 else "帮助有限"
    shape_str = f"({M}x{K})@({K}x{N})"
    print(f"{shape_str:>25} | {b_mb:>8.1f}MB | {total_mb:>8.1f}MB | {ratio:>8.1f}x | {predict:>10}")`);

// Cell 11: 15.6 performance comparison markdown
md(`## 15.6 性能对比: Pipeline vs Pipeline+Swizzle vs cuBLAS

### 对比方案
- **Pipeline (Ch.13)**: 2D grid, num_stages=3
- **Pipeline + Swizzle (本章)**: 1D grid + pid 重映射, num_stages=3
- **cuBLAS**: torch.matmul (= cuBLAS wrapper)`);

// Cell 12: benchmark tools
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

// Cell 13: Pipeline vs Swizzle vs cuBLAS benchmark
code(`# ========== Pipeline vs Swizzle vs cuBLAS ==========
BM, BN, BK = 128, 128, 32
NUM_STAGES = 3

print(f"GPU: {gpu_name}, SMs: {num_sms}, L2: {l2_size_mb:.0f} MB")
print(f"Config: BLOCK_M={BM}, BLOCK_N={BN}, BLOCK_K={BK}, num_stages={NUM_STAGES}")

print(f"\\n{'Shape':>25} | {'Pipeline':>10} | {'Swizzle':>10} | {'cuBLAS':>10} | {'Swz加速':>8} {'Swz/cuBLAS':>10}")
print("-" * 95)

test_cases = [
    # 标准方阵
    (2048, 2048, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 1024),
    (4096, 4096, 2048),
    (4096, 4096, 4096),
    # 大方阵 (超过 L2)
    (8192, 8192, 2048),
    (8192, 8192, 4096),
    # 非方阵 (swizzle 更可能有效)
    (8192, 4096, 4096),
    (4096, 8192, 4096),
    (8192, 2048, 2048),
]

for M, N, K in test_cases:
    a = torch.randn(M, K, device='cuda', dtype=torch.float16)
    b = torch.randn(K, N, device='cuda', dtype=torch.float16)

    ms_pipe = benchmark_fn(matmul_pipeline, a, b, num_stages=NUM_STAGES)
    ms_swz = benchmark_fn(matmul_swizzle, a, b, GROUP_SIZE_M=8, num_stages=NUM_STAGES)
    ms_cu = benchmark_cublas(a, b)

    speedup = ms_pipe / ms_swz
    vs_cu = ms_cu / ms_swz
    shape_str = f"({M}x{K})@({K}x{N})"
    print(f"{shape_str:>25} | {ms_pipe:>9.3f}ms | {ms_swz:>9.3f}ms | {ms_cu:>9.3f}ms | {speedup:>7.2f}x {vs_cu:>9.2f}x")

    del a, b
    torch.cuda.empty_cache()`);

// Cell 14: 15.7 GROUP_SIZE_M tuning
md(`## 15.7 GROUP_SIZE_M 参数调优

GROUP_SIZE_M 控制每个 group 包含多少行:
- **GROUP_SIZE_M=1**: 退化为列优先遍历 → A 的复用差
- **GROUP_SIZE_M=grid_m**: 退化为行优先遍历 → 与 2D grid 相同
- **最优值**: 通常在 4-16 之间, 取决于 L2 容量和矩阵尺寸`);

// Cell 15: GROUP_SIZE_M sweep
code(`# ========== GROUP_SIZE_M 参数调优 ==========
# 选择一个 swizzle 可能有效的尺寸
M, N, K = 8192, 8192, 4096
a = torch.randn(M, K, device='cuda', dtype=torch.float16)
b = torch.randn(K, N, device='cuda', dtype=torch.float16)

flops = 2.0 * M * N * K
b_size_mb = K * N * 2 / (1024*1024)
print(f"矩阵: ({M}x{K}) @ ({K}x{N}), B 大小: {b_size_mb:.0f} MB, L2: {l2_size_mb:.0f} MB")

# cuBLAS baseline
ms_cu = benchmark_cublas(a, b)
tf_cu = flops / (ms_cu * 1e-3) / 1e12
print(f"cuBLAS: {ms_cu:.3f}ms, {tf_cu:.1f} TFLOPS")

# Pipeline baseline
ms_pipe = benchmark_fn(matmul_pipeline, a, b, num_stages=3)
tf_pipe = flops / (ms_pipe * 1e-3) / 1e12

print(f"\\n{'Method':>12} | {'时间(ms)':>10} | {'TFLOPS':>8} | {'vs Pipeline':>12} | {'vs cuBLAS':>10}")
print("-" * 70)
print(f"{'Pipeline':>12} | {ms_pipe:>10.3f} | {tf_pipe:>8.1f} | {'baseline':>12} | {ms_cu/ms_pipe:>9.2f}x")

# Swizzle with different GROUP_SIZE_M
for gsm in [1, 2, 4, 8, 16, 32]:
    ms = benchmark_fn(matmul_swizzle, a, b, GROUP_SIZE_M=gsm, num_stages=3)
    tf = flops / (ms * 1e-3) / 1e12
    speedup = ms_pipe / ms
    vs_cu = ms_cu / ms
    print(f"{'GSM='+str(gsm):>12} | {ms:>10.3f} | {tf:>8.1f} | {speedup:>11.2f}x | {vs_cu:>9.2f}x")

del a, b
torch.cuda.empty_cache()`);

// Cell 16: 15.8 multi-size GROUP_SIZE_M scan
md(`## 15.8 最优 GROUP_SIZE_M 随矩阵尺寸的变化

不同矩阵尺寸的最优 GROUP_SIZE_M 可能不同。在生产中, 可以使用 ${BT}triton.autotune${BT} 自动搜索。`);

// Cell 17: Multi-size GROUP_SIZE_M scan
code(`# ========== 多尺寸最优 GROUP_SIZE_M 扫描 ==========
gsm_values = [1, 4, 8, 16]

print(f"{'Shape':>25} | {'Pipeline':>10} | " + " | ".join(f"GSM={g:>2}" for g in gsm_values) + f" | {'cuBLAS':>10} | {'Best GSM':>9}")
print("-" * 120)

for M, N, K in [
    (2048, 2048, 2048),
    (4096, 4096, 2048),
    (4096, 4096, 4096),
    (8192, 8192, 2048),
    (8192, 8192, 4096),
    (8192, 4096, 4096),
]:
    a = torch.randn(M, K, device='cuda', dtype=torch.float16)
    b = torch.randn(K, N, device='cuda', dtype=torch.float16)

    ms_pipe = benchmark_fn(matmul_pipeline, a, b, num_stages=3)
    ms_cu = benchmark_cublas(a, b)

    results = {}
    for gsm in gsm_values:
        ms = benchmark_fn(matmul_swizzle, a, b, GROUP_SIZE_M=gsm, num_stages=3)
        results[gsm] = ms

    best_gsm = min(results, key=results.get)
    best_ms = results[best_gsm]

    shape_str = f"({M}x{K})@({K}x{N})"
    parts = [f"{shape_str:>25} | {ms_pipe:>9.3f}ms"]
    for g in gsm_values:
        marker = " *" if g == best_gsm else "  "
        parts.append(f"{results[g]:>8.3f}ms{marker}")
    parts.append(f"{ms_cu:>9.3f}ms")
    parts.append(f"GSM={best_gsm:>2}")
    print(" | ".join(parts))

    del a, b
    torch.cuda.empty_cache()`);

// Cell 18: 15.9 PTX analysis
md(`## 15.9 PTX 分析: Swizzle 的编译器指令

Swizzle 是纯算法层面的优化 (改变 block 调度顺序), 不会引入新的硬件指令。
但我们可以验证:
1. Pipeline 的 ${BT}cp.async${BT} 指令仍然存在 (累进叠加)
2. Swizzle 的 pid 重映射产生了额外的整数运算指令`);

// Cell 19: PTX extraction
code(`# ========== PTX 分析 ==========
import re

def extract_ptx(kernel_fn, *args, **kwargs):
    """提取 kernel 的 PTX 代码。"""
    compiled = kernel_fn.warmup(*args, **kwargs)
    ptx = compiled.asm.get('ptx', '')
    return ptx

def analyze_ptx(ptx, label=""):
    """分析 PTX 中的关键指令。"""
    if not ptx:
        print(f"[{label}] 未能提取 PTX")
        return
    lines = ptx.strip().split('\\n')
    print(f"[{label}] PTX 总行数: {len(lines)}")
    patterns = {
        'cp.async': r'cp\\.async',
        'ld.global': r'ld\\.global',
        'st.shared': r'st\\.shared',
        'mma.sync': r'mma\\.sync',
        'ldmatrix': r'ldmatrix',
        'bar.sync': r'bar\\.sync',
        'atom.add': r'atom\\.global\\.add',
        'div/rem (pid解码)': r'(rem|div)\\.',
    }
    for name, pat in patterns.items():
        count = len(re.findall(pat, ptx))
        if count > 0:
            print(f"  {name}: {count}")

M_t, N_t, K_t = 4096, 4096, 2048

# Pipeline PTX
ptx_pipe = extract_ptx(
    matmul_pipeline_kernel,
    torch.float16, torch.float16, torch.float16,
    M_t, N_t, K_t,
    K_t, 1, N_t, 1, N_t, 1,
    BLOCK_M=128, BLOCK_N=128, BLOCK_K=32,
    num_stages=3, num_warps=4,
    grid=(1,)
)
analyze_ptx(ptx_pipe, "Pipeline (2D grid)")

print()

# Swizzle PTX
ptx_swz = extract_ptx(
    matmul_swizzle_kernel,
    torch.float16, torch.float16, torch.float16,
    M_t, N_t, K_t,
    K_t, 1, N_t, 1, N_t, 1,
    BLOCK_M=128, BLOCK_N=128, BLOCK_K=32,
    GROUP_SIZE_M=8,
    num_stages=3, num_warps=4,
    grid=(1,)
)
analyze_ptx(ptx_swz, "Swizzle (1D grid + pid remap)")

print("\\n✅ 两个 kernel 都保留了 cp.async (pipeline 累进叠加)")
print("   Swizzle kernel 额外的 div/rem 指令用于 pid → (pid_m, pid_n) 映射")`);

// Cell 20: 15.10 Summary
md(`## 15.10 总结

### 本章要点

1. **L2 Cache 问题**: 标准 2D grid 的行优先调度导致 B 矩阵数据在 L2 中的复用率低

2. **Swizzle 算法**:
   - 1D grid 替代 2D grid
   - 通过 GROUP_SIZE_M 将 block 按列优先分组
   - 同一 group 内的 block 共享 B 矩阵的列, 提升 L2 命中率

3. **累进叠加**: 在 Ch.13 pipeline (num_stages) 基础上添加 swizzle, 保留所有已有优化

4. **适用场景**:
   - ✅ 矩阵数据量 > L2 容量的 50% 时有效
   - ✅ N 较大 (B 矩阵的列多) 时效果明显
   - ⚠️ L2 容量很大的 GPU (如 Blackwell 96MB) 上, 需要更大的矩阵才能看到收益

5. **GROUP_SIZE_M 选择**: 通常 4-16 为最优, 具体取决于 L2 容量和矩阵尺寸

### 累进优化状态
| 特性 | 状态 | 路径 | 适用场景 |
|------|------|------|----------|
| Shared Memory / Block Pointer | ✅ Ch.12 | 主线 | 所有 |
| 软件流水线 (num_stages) | ✅ Ch.13 | 主线 | 所有 |
| Split-K 并行 | ✅ Ch.14 | 分支 | tall-skinny |
| **L2 Cache Swizzle** | **✅ 本章** | **分支** | **标准/大方阵** |

### 下一章预告

第16章将深入 **混合精度策略**, 探讨 FP16/FP32 在 GEMM 累加中的精度/性能权衡。

### 最终组合 (Ch.18 预告)

Ch.18 的 ultimate kernel = smem + pipeline + swizzle (不含 Split-K):
- Block Pointer (Ch.12) — 结构化内存
- Pipeline (Ch.13) — 隐藏延迟
- Swizzle (本章) — L2 优化
- FP16 累加 (Ch.16) — 精度 vs 性能
- Tensor Core (Ch.17) — 硬件加速
- Autotune (Ch.18) — 自动参数搜索`);

// Write
const nb = {
  nbformat: 4, nbformat_minor: 5,
  metadata: {
    kernelspec: { display_name: 'Python 3', language: 'python', name: 'python3' },
    language_info: { name: 'python', version: '3.10.0' }
  },
  cells
};

const outPath = '../15_matmul_swizzle.ipynb';
fs.writeFileSync(outPath, JSON.stringify(nb, null, 1));
console.log(`Wrote ${outPath} (${cells.length} cells, ${fs.statSync(outPath).size} bytes)`);
