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
md(`# 第13章：软件流水线 -- 隐藏内存延迟

## 前置知识
- 第09章：分块矩阵乘法基础
- 第12章：Block Pointer 与 Shared Memory

## 学习目标
- 理解 **软件流水线 (Software Pipelining)** 的核心思想
- 理解 CUDA 中双缓冲 / 三缓冲的设计 (${BT}simt_pipline.cu${BT}, ${BT}mma_pipline.cu${BT})
- 掌握 Triton 中 ${BT}num_stages${BT} 参数的作用
- 通过 **PTX 分析** 验证编译器生成了 ${BT}cp.async${BT} 异步拷贝指令
- 实现不同 pipeline 深度的 GEMM 并对比性能
- 与 **cuBLAS** 进行性能对比

## 对应 CUDA 代码
- ${BT}src/simt/04simt_pipline.cu${BT} — SIMT 双缓冲流水线
- ${BT}src/mma/04mma_pipline.cu${BT} — MMA 双缓冲流水线
- 核心思想：加载下一个 tile 的同时计算当前 tile

## 累进优化
- ✅ Shared Memory / Block Pointer (Ch.12)
- ✅ **软件流水线 (本章新增)**`);

// ============================================================
// Cell 1: imports
// ============================================================
code(`import torch
import triton
import triton.language as tl
import re, collections`);

// ============================================================
// Cell 2: 13.1 内存延迟问题
// ============================================================
md(`## 13.1 内存延迟问题

### GPU 内存访问延迟

${BT3}
访问类型           延迟 (cycles)   带宽
────────────────────────────────────────────
Register           ~1              —
Shared Memory      ~20-30          ~TB/s
L2 Cache           ~200-300        ~2-4 TB/s
Global Memory      ~400-600        ~1-2 TB/s
${BT3}

### 无流水线的 GEMM (停顿问题)

${BT3}
时间线 (无流水线, num_stages=1):

K=0:    |███ Load A0,B0 ███|██ Compute ██|
K=BK:   |                  |             |███ Load A1,B1 ███|██ Compute ██|
K=2BK:  |                  |             |                  |             |███ Load ███|██ Compute ██|

        ▼ 计算单元空闲 ▼   ▼ 内存空闲 ▼  ▼ 计算单元空闲 ▼   ▼ 内存空闲 ▼

问题: 加载和计算串行执行 → 总时间 = 加载时间 + 计算时间
${BT3}

### 流水线的解决方案

${BT3}
时间线 (双缓冲流水线, num_stages=2):

加载:   |███ Load A0,B0 ███|███ Load A1,B1 ███|███ Load A2,B2 ███|
计算:   |                  |██ Compute(A0,B0)██|██ Compute(A1,B1)██|█ Comp(A2,B2)█|

        ↑ 预加载第一个 tile  ↑ 加载和计算重叠!

关键: 当计算 tile k 时, 同时加载 tile k+1
  → 总时间 ≈ max(加载时间, 计算时间) (理想情况)
${BT3}`);

// ============================================================
// Cell 3: 13.2 CUDA Pipeline 回顾
// ============================================================
md(`## 13.2 CUDA Pipeline 的实现回顾

### simt_pipline.cu 的流水线结构

${BT3}
 ←-----------------------------------------------------------------------------------
 ⤷---------------------------------------iter k-----------------------------------→-⤴
 |████████████████load global███████████████████████|███store shared███|             |
 |---------------------------------------iter bk-----------------------↘-------------|
 |█load shared█|█load shared█|█load shared█|█load shared█|█load shared█|█load shared█|
 ↘-------------↘------------↘-------------↘-------------↘-------------↘-------------↘
 |████Math█████|████Math█████|████Math█████|████Math█████|████Math█████|████Math█████|
${BT3}

### CUDA 双缓冲 (Double Buffering) 的代码复杂度

${BT3}cpp
// CUDA: 需要手动管理双缓冲 (~220 行)
__shared__ float4 smem_A[2][...];   // 两个 buffer
__shared__ float4 smem_B[2][...];
float4 reg_a[2][...];               // 两个寄存器组
float4 reg_b[2][...];

// Prologue: 预加载第一个 tile
LOAD_GLOBAL(0);
STORE_SHARED(0);
__syncthreads();
LOAD_SHARED(0, 0, 0);

// Main loop: 手动交替 read/write buffer
for (int k = 1; k < K/BK; k++) {
    smem_write_idx = !smem_write_idx;  // 切换 buffer
    LOAD_GLOBAL(k);
    // ... 复杂的 prologue/epilogue 处理
}
${BT3}

### Triton 的等效代码

${BT3}python
# Triton: 整个流水线只需改一个参数!
for k in range(0, K, BLOCK_K):
    a_tile = tl.load(a_block_ptr, boundary_check=(0, 1))
    b_tile = tl.load(b_block_ptr, boundary_check=(0, 1))
    acc = tl.dot(a_tile, b_tile, acc=acc)
    a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
    b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))
# + num_stages=3 在 launch 时指定
${BT3}

| 方面 | CUDA (simt_pipline) | Triton |
|------|--------------------|---------|
| 代码行数 | ~220 行 | ~10 行 |
| 缓冲管理 | 手动 smem[2], reg[2] | 编译器自动 |
| 索引切换 | idx = !idx | 编译器自动 |
| Prologue/Epilogue | 手动特殊处理 | 编译器自动 |
| 异步拷贝 | 手动 cp.async | 编译器自动 |
| 流水线深度 | 硬编码 (双缓冲) | 运行时可调 |`);

// ============================================================
// Cell 4: 13.3 num_stages 含义
// ============================================================
md(`## 13.3 Triton 的流水线: num_stages 参数

### num_stages 的含义

${BT3}
num_stages = 1: 无流水线
  加载 → 计算 → 加载 → 计算 → ...
  SMEM 使用: 1 × (BM×BK + BK×BN) × sizeof(half)

num_stages = 2: 双缓冲 (对应 CUDA 的 smem_A[2])
  加载 tile 1 → 同时 {加载 tile 2, 计算 tile 1} → ...
  SMEM 使用: 2 × (BM×BK + BK×BN) × sizeof(half)

num_stages = 3: 三缓冲
  预加载 tile 1,2 → 同时 {加载 tile 3, 计算 tile 1} → ...
  SMEM 使用: 3 × (BM×BK + BK×BN) × sizeof(half)

num_stages = 4: 四缓冲
  SMEM 使用: 4 × (BM×BK + BK×BN) × sizeof(half)
${BT3}

### 编译器生成的指令

${BT3}
num_stages > 1 时, 编译器会在 Ampere+ GPU 上生成:

1. cp.async.ca.shared.global  (异步全局→共享内存拷贝)
   - 不阻塞计算管线
   - 硬件 DMA 引擎执行数据搬运

2. cp.async.commit_group      (提交一组异步拷贝)

3. cp.async.wait_group N      (等待第 N 组之前的拷贝完成)
   - N=0: 等待所有拷贝完成
   - N=1: 允许最后1组拷贝还在进行中
${BT3}

### Shared Memory 使用量

${BT3}
以 BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, FP16 为例:

每个 stage 需要:
  A tile: 128 × 32 × 2 bytes = 8 KB
  B tile: 32 × 128 × 2 bytes = 8 KB
  每 stage: 16 KB

num_stages=1: 16 KB
num_stages=2: 32 KB
num_stages=3: 48 KB
num_stages=4: 64 KB  (接近 SM 的 smem 上限)
num_stages=5: 80 KB  (可能超出某些 GPU 的限制!)
${BT3}`);

// ============================================================
// Cell 5: 13.4 实现 markdown
// ============================================================
md(`## 13.4 实现: Pipeline GEMM Kernel

在 Ch.12 的 Block Pointer kernel 基础上，**只改变一个参数** — ${BT}num_stages${BT}。

Kernel 代码与 Ch.12 完全相同，流水线由编译器根据 ${BT}num_stages${BT} 自动插入。`);

// ============================================================
// Cell 6: Pipeline kernel
// ============================================================
code(`@triton.jit
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
    """
    Pipeline GEMM kernel — 在 Ch.12 Block Pointer 基础上添加流水线。

    累进优化:
      ✅ make_block_ptr (Ch.12) — 结构化内存访问
      ✅ FP32 累加器 — 数值精度
      ✅ Tensor Core (tl.dot) — 自动映射
      ✅ num_stages (本章) — 编译器自动插入流水线

    对应 CUDA simt_pipline.cu / mma_pipline.cu:
    - CUDA 需要 ~220 行手动管理双缓冲
    - Triton 只需在 launch 时指定 num_stages
    """
    # 2D grid (与 Ch.12 相同, 此时还没有 swizzle)
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Block Pointer (Ch.12)
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

    # FP32 累加器
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # K 循环 — 编译器根据 num_stages 自动生成流水线代码
    for k in range(0, K, BLOCK_K):
        a_tile = tl.load(a_block_ptr, boundary_check=(0, 1))
        b_tile = tl.load(b_block_ptr, boundary_check=(0, 1))
        acc = tl.dot(a_tile, b_tile, acc=acc)
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))

    # 写回
    c_block_ptr = tl.make_block_ptr(
        base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N), order=(1, 0),
    )
    tl.store(c_block_ptr, acc.to(tl.float16), boundary_check=(0, 1))`);

// ============================================================
// Cell 7: Host wrapper
// ============================================================
code(`def matmul_pipeline(a, b, BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, num_stages=2):
    """Pipeline GEMM 的 host 端包装函数。"""
    assert a.dtype == torch.float16 and b.dtype == torch.float16
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

// ============================================================
// Cell 8: 正确性验证
// ============================================================
code(`# ========== 正确性验证 ==========
torch.manual_seed(42)
M, N, K = 2048, 2048, 1024
a = torch.randn(M, K, device='cuda', dtype=torch.float16)
b = torch.randn(K, N, device='cuda', dtype=torch.float16)
c_ref = torch.matmul(a, b)

print("不同 num_stages 的正确性验证:")
for stages in [1, 2, 3, 4]:
    c_tri = matmul_pipeline(a, b, num_stages=stages)
    max_err = (c_tri - c_ref).abs().max().item()
    rel_err = torch.norm(c_tri.float() - c_ref.float()) / torch.norm(c_ref.float())
    print(f"  num_stages={stages}: max_err={max_err:.4f}, rel_err={rel_err:.6f}, "
          f"correct={torch.allclose(c_tri, c_ref, atol=1.0)}")`);

// ============================================================
// Cell 9: 13.5 PTX 分析 markdown
// ============================================================
md(`## 13.5 PTX 分析: 验证 cp.async 指令

这是本章最重要的验证: 通过对比 ${BT}num_stages=1${BT} 和 ${BT}num_stages=3${BT} 的 PTX,
确认编译器是否真的插入了异步拷贝指令。

**预期**:
- ${BT}num_stages=1${BT}: 使用 ${BT}ld.global${BT} 同步加载, 无 ${BT}cp.async${BT}
- ${BT}num_stages=3${BT}: 使用 ${BT}cp.async.cg.shared.global${BT} 异步加载, 有 ${BT}commit_group${BT} / ${BT}wait_group${BT}`);

// ============================================================
// Cell 10: PTX analysis code
// ============================================================
code(`# ========== PTX 分析: num_stages=1 vs num_stages=3 ==========
def extract_ptx(kernel_fn, *args, **kwargs):
    """从 Triton kernel 编译结果中提取 PTX。"""
    compiled = kernel_fn.warmup(*args, **kwargs)
    ptx = compiled.asm.get('ptx', '')
    if not ptx:
        for key in compiled.asm:
            if 'ptx' in key.lower():
                ptx = compiled.asm[key]
                break
    return ptx

def analyze_ptx(ptx, label=""):
    """分析 PTX 中的关键 GPU 指令。"""
    patterns = {
        'cp.async':    r'cp\.async[\.\w]+',
        'ld.global':   r'ld\.global[\.\w]+',
        'st.shared':   r'st\.shared[\.\w]+',
        'ld.shared':   r'ld\.shared[\.\w]+',
        'mma.sync':    r'mma\.sync[\.\w]+',
        'ldmatrix':    r'ldmatrix[\.\w]+',
        'bar.sync':    r'bar[\.\w]+',
    }
    print(f"\\n{'='*60}")
    print(f"  PTX 指令分析: {label}")
    print(f"{'='*60}")

    if not ptx:
        print("  (未能提取 PTX)")
        return {}

    counts = {}
    for name, pat in patterns.items():
        matches = re.findall(pat, ptx)
        counts[name] = len(matches)
        if matches:
            unique = collections.Counter(matches).most_common(3)
            detail = ', '.join(f'{k}: {v}' for k, v in unique)
            print(f"  {name:>12}: {len(matches):>3} 条  ({detail})")
        else:
            print(f"  {name:>12}:   0 条")

    print(f"  PTX 总长度: {len(ptx)} 字节, ~{len(ptx.splitlines())} 行")
    return counts

# 提取两种配置的 PTX
ptx_args = (
    torch.float16, torch.float16, torch.float16,
    2048, 2048, 1024,
    1024, 1,   # stride_am, stride_ak
    2048, 1,   # stride_bk, stride_bn
    2048, 1,   # stride_cm, stride_cn
)
ptx_kwargs_base = dict(BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, num_warps=4)

try:
    ptx_s1 = extract_ptx(matmul_pipeline_kernel, *ptx_args,
                          **ptx_kwargs_base, num_stages=1, grid=(16, 16))
    counts_s1 = analyze_ptx(ptx_s1, "num_stages=1 (无流水线)")

    ptx_s3 = extract_ptx(matmul_pipeline_kernel, *ptx_args,
                          **ptx_kwargs_base, num_stages=3, grid=(16, 16))
    counts_s3 = analyze_ptx(ptx_s3, "num_stages=3 (三级流水线)")

    print("\\n" + "="*60)
    print("  关键对比")
    print("="*60)
    print(f"  {'指令':>12} | {'stages=1':>10} | {'stages=3':>10} | 变化")
    print(f"  {'-'*50}")
    for key in ['cp.async', 'ld.global', 'mma.sync', 'ldmatrix']:
        v1 = counts_s1.get(key, 0)
        v3 = counts_s3.get(key, 0)
        diff = v3 - v1
        sign = '+' if diff > 0 else ''
        print(f"  {key:>12} | {v1:>10} | {v3:>10} | {sign}{diff}")

    print("\\n关键发现:")
    if counts_s3.get('cp.async', 0) > counts_s1.get('cp.async', 0):
        print("  ✓ num_stages=3 使用了 cp.async 异步拷贝 — 流水线生效!")
    elif counts_s3.get('cp.async', 0) > 0:
        print("  ✓ 两者都使用了 cp.async (编译器可能默认启用)")
    else:
        print("  △ 未检测到 cp.async — 可能使用了其他异步机制或 GPU 不支持")
    if counts_s3.get('ld.global', 0) < counts_s1.get('ld.global', 0):
        print("  ✓ num_stages=3 减少了 ld.global 同步加载")

except Exception as e:
    print(f"PTX 提取失败: {e}")
    print("注: 不同 Triton 版本的 warmup API 可能有差异")`);

// ============================================================
// Cell 11: 13.6 性能对比 markdown
// ============================================================
md(`## 13.6 性能对比: 不同 num_stages

接下来对比不同 ${BT}num_stages${BT} 的性能, 并与 **cuBLAS** 基准进行比较。`);

// ============================================================
// Cell 12: benchmark_vs_cublas + num_stages comparison
// ============================================================
code(`# ========== benchmark 工具函数 ==========
def benchmark_vs_cublas(triton_fn, M, N, K, num_warmup=25, num_rep=100, **kwargs):
    """统一的 Triton vs cuBLAS (torch.matmul) benchmark。"""
    a = torch.randn(M, K, device='cuda', dtype=torch.float16)
    b = torch.randn(K, N, device='cuda', dtype=torch.float16)
    flops = 2.0 * M * N * K

    # Triton
    for _ in range(num_warmup):
        triton_fn(a, b, **kwargs)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(num_rep):
        triton_fn(a, b, **kwargs)
    e.record()
    torch.cuda.synchronize()
    ms_tri = s.elapsed_time(e) / num_rep

    # cuBLAS
    for _ in range(num_warmup):
        torch.matmul(a, b)
    torch.cuda.synchronize()
    s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    s.record()
    for _ in range(num_rep):
        torch.matmul(a, b)
    e.record()
    torch.cuda.synchronize()
    ms_cu = s.elapsed_time(e) / num_rep

    tflops_tri = flops / (ms_tri * 1e-3) / 1e12
    tflops_cu = flops / (ms_cu * 1e-3) / 1e12
    return ms_tri, ms_cu, tflops_tri, tflops_cu`);

// ============================================================
// Cell 13: num_stages sweep
// ============================================================
code(`# ========== 不同 num_stages 的性能 ==========
M, N, K = 2048, 2048, 1024
BM, BN, BK = 128, 128, 32

print(f"矩阵规模: M={M}, N={N}, K={K}")
print(f"Block: ({BM}, {BN}, {BK})")
print(f"每 stage SMEM 使用: {(BM*BK + BK*BN)*2/1024:.1f} KB")

# 先获取 cuBLAS 基准
a = torch.randn(M, K, device='cuda', dtype=torch.float16)
b = torch.randn(K, N, device='cuda', dtype=torch.float16)
_, ms_cublas, _, tflops_cublas = benchmark_vs_cublas(matmul_pipeline, M, N, K, num_stages=2)

print(f"cuBLAS 基准: {ms_cublas:.3f}ms, {tflops_cublas:.1f} TFLOPS")
print(f"\\n{'num_stages':>12} | {'SMEM(KB)':>10} | {'时间(ms)':>10} | {'TFLOPS':>8} | {'vs cuBLAS':>10}")
print("-" * 65)

for stages in [1, 2, 3, 4, 5]:
    smem_kb = stages * (BM * BK + BK * BN) * 2 / 1024
    try:
        ms_t, ms_c, tf_t, tf_c = benchmark_vs_cublas(
            matmul_pipeline, M, N, K,
            BLOCK_M=BM, BLOCK_N=BN, BLOCK_K=BK, num_stages=stages)
        eff = ms_c / ms_t
        print(f"{stages:>12} | {smem_kb:>10.1f} | {ms_t:>10.3f} | {tf_t:>8.1f} | {eff:>9.0%}")
    except Exception as e:
        print(f"{stages:>12} | {smem_kb:>10.1f} | {'FAIL':>10} | — | (SMEM 超出上限?)")`);

// ============================================================
// Cell 14: Different matrix sizes
// ============================================================
code(`# ========== 不同矩阵尺寸下 pipeline 的效果 ==========
print("不同矩阵尺寸下 num_stages 的影响 (vs cuBLAS)")
print(f"{'Size':>20} | {'stage=1':>12} {'stage=2':>12} {'stage=3':>12} | {'cuBLAS':>12} | {'最优stage':>10}")
print("-" * 95)

for M, N, K in [
    (1024, 1024, 1024),
    (2048, 2048, 1024),
    (2048, 2048, 2048),
    (4096, 4096, 1024),
    (4096, 4096, 2048),
    (4096, 4096, 4096),
]:
    results = []
    ms_cu = None
    for stages in [1, 2, 3]:
        try:
            ms_t, mc, _, _ = benchmark_vs_cublas(
                matmul_pipeline, M, N, K,
                BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, num_stages=stages)
            results.append((stages, ms_t))
            ms_cu = mc
        except:
            results.append((stages, float('inf')))

    best = min(results, key=lambda x: x[1])
    size_str = f"{M}x{N}x{K}"
    ms_strs = [f"{r[1]:>11.3f}ms" if r[1] < float('inf') else f"{'FAIL':>12}" for r in results]
    cu_str = f"{ms_cu:>11.3f}ms" if ms_cu else "N/A"
    print(f"{size_str:>20} | {ms_strs[0]} {ms_strs[1]} {ms_strs[2]} | {cu_str} | {'stage='+str(best[0]):>10}")`);

// ============================================================
// Cell 15: 13.7 BLOCK_K × num_stages 交互
// ============================================================
md(`## 13.7 BLOCK_K × num_stages 的交互效应

更大的 ${BT}BLOCK_K${BT} 可以提高算术强度 (减少 K 循环次数), 但也增加每个 stage 的 SMEM 使用量。
两个参数需要联合调优。`);

// ============================================================
// Cell 16: BLOCK_K × num_stages grid search
// ============================================================
code(`# ========== BLOCK_K × num_stages 交叉实验 ==========
M, N, K = 2048, 2048, 2048

print(f"BLOCK_K × num_stages 交互 (M={M}, N={N}, K={K})")
print(f"{'BLOCK_K':>8} {'stages':>8} | {'SMEM(KB)':>10} {'时间(ms)':>10} {'TFLOPS':>8} | {'vs cuBLAS':>10}")
print("-" * 70)

# cuBLAS baseline
a = torch.randn(M, K, device='cuda', dtype=torch.float16)
b = torch.randn(K, N, device='cuda', dtype=torch.float16)
for _ in range(25):
    torch.matmul(a, b)
torch.cuda.synchronize()
s, e = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
s.record()
for _ in range(100):
    torch.matmul(a, b)
e.record()
torch.cuda.synchronize()
ms_cublas = s.elapsed_time(e) / 100
flops = 2.0 * M * N * K
tf_cublas = flops / (ms_cublas * 1e-3) / 1e12
print(f"{'cuBLAS':>17} | {'—':>10} {ms_cublas:>10.3f} {tf_cublas:>8.1f} | {'baseline':>10}")
print("-" * 70)

best_ms = float('inf')
best_config = None

for bk in [16, 32, 64]:
    for stages in [1, 2, 3, 4]:
        smem_kb = stages * (128 * bk + bk * 128) * 2 / 1024
        try:
            ms_t, _, tf_t, _ = benchmark_vs_cublas(
                matmul_pipeline, M, N, K,
                BLOCK_M=128, BLOCK_N=128, BLOCK_K=bk, num_stages=stages)
            eff = ms_cublas / ms_t
            marker = " ◄ best" if ms_t < best_ms else ""
            if ms_t < best_ms:
                best_ms = ms_t
                best_config = (bk, stages)
            print(f"{bk:>8} {stages:>8} | {smem_kb:>10.1f} {ms_t:>10.3f} {tf_t:>8.1f} | {eff:>9.0%}{marker}")
        except Exception as e:
            print(f"{bk:>8} {stages:>8} | {smem_kb:>10.1f} {'FAIL':>10} {'—':>8} | (SMEM 超出上限)")

if best_config:
    print(f"\\n最优配置: BLOCK_K={best_config[0]}, num_stages={best_config[1]} → {best_ms:.3f}ms")`);

// ============================================================
// Cell 17: 13.8 总结
// ============================================================
md(`## 13.8 总结

### 本章要点

1. **软件流水线的核心思想**：当计算当前 tile 时，同时加载下一个 tile，重叠延迟

2. **CUDA vs Triton**：
   - CUDA 需要手动实现双缓冲 (smem[2], reg[2])，手动管理索引切换，手动处理 prologue/epilogue
   - Triton 只需指定 ${BT}num_stages${BT} 参数，编译器自动完成所有工作

3. **PTX 验证**：
   - ${BT}num_stages=1${BT}: 使用 ${BT}ld.global${BT} 同步加载
   - ${BT}num_stages≥2${BT}: 使用 ${BT}cp.async.cg.shared.global${BT} 异步加载 + ${BT}commit_group${BT} / ${BT}wait_group${BT}
   - 通过 PTX 分析**证实**编译器确实生成了流水线指令

4. **num_stages 的选择**：
   - ${BT}1${BT}: 无流水线，最少 SMEM
   - ${BT}2${BT}: 双缓冲，最常用
   - ${BT}3-4${BT}: 更深的流水线，可能更好地隐藏延迟，但需要更多 SMEM
   - 受 Shared Memory 容量限制

5. **BLOCK_K × num_stages 联合调优**：
   - 更大的 BLOCK_K 提高算术强度，但增加 SMEM 使用
   - 需要联合搜索找到最优组合

### 累进优化状态
| 特性 | 状态 |
|------|------|
| Shared Memory / Block Pointer | ✅ Ch.12 |
| **软件流水线 (num_stages)** | **✅ 本章** |
| Split-K 并行 | → Ch.14 |
| L2 Cache Swizzle | → Ch.15 |

### 练习

1. **PTX 深入**：在 PTX 中找到 ${BT}cp.async.wait_group${BT} 指令, 分析 wait 的 group 数量与 num_stages 的关系
2. **K 维度影响**：固定 M=N=2048, 改变 K (256, 512, 1024, 2048, 4096), 观察不同 num_stages 的效果
3. **SMEM 上限实验**：增大 BLOCK_M/BLOCK_N 直到 num_stages=2 也无法编译, 观察报错信息
4. **思考题**：为什么在 K 很小时 (如 K=64), 流水线可能反而更慢? (提示: prologue 开销)

### 下一章预告

第14章将介绍 **Split-K 并行**, 通过在 K 维度引入并行性来加速 tall-skinny 矩阵乘法。
Split-K 将在本章 pipeline kernel 的基础上累进叠加。`);

// ============================================================
// Write notebook
// ============================================================
const nb = {
  nbformat: 4,
  nbformat_minor: 5,
  metadata: {
    kernelspec: { display_name: 'Python 3', language: 'python', name: 'python3' },
    language_info: { name: 'python', version: '3.10.0' }
  },
  cells
};

const outPath = '../13_matmul_pipeline.ipynb';
fs.writeFileSync(outPath, JSON.stringify(nb, null, 1));
console.log(`Wrote ${outPath} (${cells.length} cells, ${fs.statSync(outPath).size} bytes)`);
