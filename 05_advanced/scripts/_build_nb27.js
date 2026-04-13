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
md(`# 第27章：跨平台视角 — PyPTO vs Triton 矩阵乘法对比

## 前置知识
- 第09章：分块矩阵乘法
- 第17章：Tensor Core 与 tl.dot 映射
- 第18章：终极优化 GEMM
- 了解华为昇腾 NPU 基本概念（本章会介绍）

## 学习目标
- 理解华为 **PyPTO** 框架的设计思想和 API
- 对比 PyPTO 与 Triton 在矩阵乘法实现上的差异
- 理解 **MPMD vs SPMD** 执行模型对 GEMM 的影响
- 掌握 PyPTO 的 **Cube Tile** 配置与 Triton Block Size 的对应关系
- 理解 **ND vs NZ** 数据格式在两个框架中的角色

## 本章性质
- PyPTO 代码为**解释性展示**（需要昇腾 NPU 环境才能运行）
- Triton 代码可在 NVIDIA GPU 上正常运行
- 重点是**概念对比**和**设计哲学**的理解`);

// ============================================================
// Cell 1: imports (Triton side only)
// ============================================================
code(`import torch
import triton
import triton.language as tl

# PyPTO 需要昇腾 NPU 环境，这里仅展示 API
# import pypto as pto  # 需要: pip install pypto + CANN 环境`);

// ============================================================
// Cell 2: 27.1 Why cross-platform comparison
// ============================================================
md(`## 27.1 为什么要做跨平台对比？

在 AI 加速器领域，NVIDIA GPU 不是唯一选择。华为昇腾 NPU 已在多个大模型（DeepSeek-V3、GLM-4.5）
中落地部署。理解不同硬件的编程范式，能让我们：

1. **抽象出通用优化原理** — Tiling、流水线、bank conflict 避免这些概念是跨平台的
2. **理解设计取舍** — 为什么 NVIDIA 选 SPMD 而华为选 MPMD？不是随意的，是硬件架构决定的
3. **扩展职业技能** — 大模型推理正在向多硬件平台迁移

### 两个框架的一句话定位

${BT3}
Triton (NVIDIA):
  "用 Python 写 GPU kernel，编译器帮你处理底层细节"
  → 面向 NVIDIA GPU 的 Block/Tile 级编程框架
  → SPMD 执行模型（一份代码，多个 block 并行）

PyPTO (华为):
  "用 Tensor API 写算子，框架帮你编译到 NPU 指令"
  → 面向昇腾 NPU 的 Tensor 级编程框架
  → MPMD 执行模型（不同核可运行不同程序）
${BT3}`);

// ============================================================
// Cell 3: 27.2 Hardware architecture
// ============================================================
md(`## 27.2 硬件架构对比

理解编程模型差异的根源，必须先看硬件。

${BT3}
NVIDIA GPU (Ampere/Hopper/Blackwell):

  ┌─────────────────────────────────────────────────┐
  │  GPU Chip                                        │
  │  ┌─────────┐  ┌─────────┐       ┌─────────┐    │
  │  │   SM 0  │  │   SM 1  │  ...  │  SM 107  │    │
  │  │┌──────┐ │  │┌──────┐ │       │┌──────┐  │    │
  │  ││CUDA  │ │  ││CUDA  │ │       ││CUDA   │ │    │
  │  ││Cores │ │  ││Cores │ │       ││Cores  │ │    │
  │  │├──────┤ │  │├──────┤ │       │├──────┤  │    │
  │  ││Tensor│ │  ││Tensor│ │       ││Tensor │ │    │
  │  ││Cores │ │  ││Cores │ │       ││Cores  │ │    │
  │  │├──────┤ │  │├──────┤ │       │├──────┤  │    │
  │  ││Shared│ │  ││Shared│ │       ││Shared │ │    │
  │  ││Memory│ │  ││Memory│ │       ││Memory │ │    │
  │  │└──────┘ │  │└──────┘ │       │└──────┘  │    │
  │  └─────────┘  └─────────┘       └─────────┘    │
  │            L2 Cache (40-60 MB)                   │
  │            HBM (80-192 GB)                       │
  └─────────────────────────────────────────────────┘

  特点: 所有 SM 结构相同 (同构), 适合 SPMD


华为昇腾 NPU (Atlas A2/A3):

  ┌─────────────────────────────────────────────────┐
  │  NPU Chip                                        │
  │  ┌──────────┐  ┌──────────┐    ┌──────────┐    │
  │  │ AI Core 0│  │ AI Core 1│    │AI Core N │    │
  │  │┌────────┐│  │┌────────┐│    │┌────────┐│    │
  │  ││ Cube   ││  ││ Cube   ││    ││ Cube   ││    │
  │  ││ Unit   ││  ││ Unit   ││    ││ Unit   ││    │
  │  │├────────┤│  │├────────┤│    │├────────┤│    │
  │  ││ Vector ││  ││ Vector ││    ││ Vector ││    │
  │  ││ Unit   ││  ││ Unit   ││    ││ Unit   ││    │
  │  │├────────┤│  │├────────┤│    │├────────┤│    │
  │  ││ Scalar ││  ││ Scalar ││    ││ Scalar ││    │
  │  ││ Unit   ││  ││ Scalar ││    ││ Unit   ││    │
  │  │├────────┤│  │├────────┤│    │├────────┤│    │
  │  ││  SRAM  ││  ││  SRAM  ││    ││  SRAM  ││    │
  │  ││(L0/L1) ││  ││(L0/L1) ││    ││(L0/L1) ││    │
  │  │└────────┘│  │└────────┘│    │└────────┘│    │
  │  └──────────┘  └──────────┘    └──────────┘    │
  │            L2 Cache                              │
  │            HBM                                   │
  └─────────────────────────────────────────────────┘

  特点:
  - Cube Unit = 矩阵乘法专用单元 (类似 Tensor Core)
  - Vector Unit = 向量运算单元 (类似 CUDA Core)
  - Scalar Unit = 标量控制单元
  - 三个单元可以独立调度 → 天然适合 MPMD
${BT3}

### 关键差异

| 维度 | NVIDIA GPU | 华为昇腾 NPU |
|------|-----------|-------------|
| 计算单元 | SM (同构) | AI Core (Cube + Vector + Scalar) |
| 矩阵乘法 | Tensor Core (warp 级) | Cube Unit (核级) |
| 向量运算 | CUDA Core | Vector Unit |
| 片上存储 | Shared Memory (可编程) | SRAM (L0/L1/UB 多级) |
| 编程粒度 | Warp (32 线程) | 整个 AI Core |
| 执行模型 | SPMD | MPMD |`);

// ============================================================
// Cell 4: 27.3 Programming model comparison
// ============================================================
md(`## 27.3 编程模型对比：SPMD vs MPMD

这是两个框架最根本的差异，直接影响 GEMM 的实现方式。

${BT3}
Triton (SPMD):
  一份 kernel 代码 → 多个 program instance 并行执行

  kernel(data, pid=0)  kernel(data, pid=1)  kernel(data, pid=2)
       │                    │                    │
       ▼                    ▼                    ▼
     SM 0                 SM 1                 SM 2
  处理 block(0,0)      处理 block(0,1)      处理 block(1,0)

  每个 program 通过 tl.program_id() 知道自己处理哪块数据
  所有 program 运行完全相同的代码


PyPTO (MPMD):
  多份不同程序 → 通过任务 DAG 调度到不同核

  Task A (搬数据)   Task B (矩阵乘)   Task C (后处理)
       │                 │                 │
       ▼                 ▼                 ▼
   AI Core 0          AI Core 1        AI Core 2
   (Vector Unit)      (Cube Unit)      (Vector Unit)

  不同核可以运行完全不同的程序
  通过任务依赖图 (DAG) 协调，避免全局同步
${BT3}

### 对 GEMM 的影响

${BT3}
Triton GEMM:
  每个 program 独立计算 C 矩阵的一个 tile
  所有 program 执行相同的 "load A tile, load B tile, dot, store" 循环
  数据复用靠 L2 cache + swizzle 访问模式

PyPTO GEMM:
  框架将 GEMM 分解为多个 task:
  - Task 1: 从 HBM 搬运 A tile 到 L1 SRAM
  - Task 2: 从 HBM 搬运 B tile 到 L1 SRAM
  - Task 3: Cube Unit 执行 matmul (SRAM → L0)
  - Task 4: 从 L0 搬运结果到 HBM
  这些 task 通过 DAG 依赖自动流水化
${BT3}`);

// ============================================================
// Cell 5: 27.4 API comparison
// ============================================================
md(`## 27.4 矩阵乘法 API 对比

### Triton 实现

在 Triton 中，矩阵乘法需要开发者手动编写 kernel，控制分块、循环、数据搬运：`);

// ============================================================
// Cell 6: Triton GEMM kernel (runnable)
// ============================================================
code(`# Triton GEMM — 开发者手动控制一切
@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    # 1. 确定当前 program 负责哪个 tile
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # 2. 创建 block pointer (编译器自动处理 smem swizzle)
    a_block_ptr = tl.make_block_ptr(
        base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak),
        offsets=(pid_m * BLOCK_M, 0),
        block_shape=(BLOCK_M, BLOCK_K), order=(1, 0),  # 行主序
    )
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn),
        offsets=(0, pid_n * BLOCK_N),
        block_shape=(BLOCK_K, BLOCK_N), order=(1, 0),
    )

    # 3. K 维循环 — 分块累加
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a_tile = tl.load(a_block_ptr, boundary_check=(0, 1))
        b_tile = tl.load(b_block_ptr, boundary_check=(0, 1))
        acc = tl.dot(a_tile, b_tile, acc=acc)  # ← Tensor Core!
        a_block_ptr = tl.advance(a_block_ptr, (0, BLOCK_K))
        b_block_ptr = tl.advance(b_block_ptr, (BLOCK_K, 0))

    # 4. 写回结果
    c_block_ptr = tl.make_block_ptr(
        base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N), order=(1, 0),
    )
    tl.store(c_block_ptr, acc.to(tl.float16), boundary_check=(0, 1))

print("Triton GEMM kernel 定义完成")`);

// ============================================================
// Cell 7: PyPTO GEMM explanation
// ============================================================
md(`### PyPTO 实现

在 PyPTO 中，矩阵乘法可以在多个抽象层级实现：

**层级 1：Tensor 级 — 一行代码**

${BT3}python
import pypto as pto

# 创建张量
a = pto.tensor([M, K], pto.DT_FP16, "A")
b = pto.tensor([K, N], pto.DT_FP16, "B")

# 矩阵乘法 — 一行搞定
c = pto.matmul(a, b, out_dtype=pto.DT_FP32)

# 等价的 Tensor 方法调用
c = a.matmul(b, out_dtype=pto.DT_FP32)
${BT3}

说白了，这就像写 PyTorch 的 ${BT}torch.matmul()${BT}。框架在背后完成所有的 tiling、数据搬运、指令选择。

**层级 2：配置 Cube Tile — 手动控制分块策略**

${BT3}python
import pypto as pto

# 配置 Cube Tile Shape（类似 Triton 的 BLOCK_M, BLOCK_N, BLOCK_K）
# 参数: [m_L0, m_L1], [k_L0, k_L1], [n_L0, n_L1]
# L0 = 最近计算的缓存（类似寄存器），L1 = 片上 SRAM（类似 shared memory）
pto.set_cube_tile_shapes(
    [128, 128],    # M 轴: L0=128, L1=128
    [256, 512],    # K 轴: L0=256, L1=512
    [128, 128],    # N 轴: L0=128, L1=128
)

# 然后执行 matmul — 框架按你指定的 tile shape 编译
c = pto.matmul(a, b, out_dtype=pto.DT_FP32)
${BT3}

**层级 3：完整 Kernel 函数 — JIT 编译**

${BT3}python
import pypto as pto

@pto.frontend.jit
def gemm_kernel(
    a: pto.Tensor,     # 输入 A [M, K]
    b: pto.Tensor,     # 输入 B [K, N]
    c: pto.Tensor,     # 输出 C [M, N]
):
    # 配置 tiling
    pto.set_cube_tile_shapes([128, 128], [256, 512], [128, 128])

    # K 维循环（框架自动展开和流水化）
    for k_idx in pto.loop(0, K // k_tile, 1, name="k_loop"):
        a_tile = a[..., k_idx*k_tile:(k_idx+1)*k_tile]
        b_tile = b[k_idx*k_tile:(k_idx+1)*k_tile, ...]
        c += pto.matmul(a_tile, b_tile, out_dtype=pto.DT_FP32)

# 调用 — 首次触发 JIT 编译
a_data = pto.from_torch(torch.randn(M, K, dtype=torch.float16, device='npu'), "A")
b_data = pto.from_torch(torch.randn(K, N, dtype=torch.float16, device='npu'), "B")
c_data = pto.tensor([M, N], pto.DT_FP32, "C")
gemm_kernel(a_data, b_data, c_data)
${BT3}`);

// ============================================================
// Cell 8: 27.5 Tiling strategy comparison
// ============================================================
md(`## 27.5 Tiling 策略对比

两个框架都用 Tiling 来将大矩阵分块到片上存储。但控制方式和层次完全不同。

${BT3}
Triton 的 Tiling (2 级):

  Global Memory ──→ Shared Memory ──→ Registers
  (HBM)              (自动管理)        (自动管理)
        ↑                  ↑
    block_ptr          编译器自动
    boundary_check     swizzle + ldmatrix

  开发者控制:
    BLOCK_M, BLOCK_N, BLOCK_K  (constexpr 参数)
    num_stages                  (流水线深度)
    order=(1,0)                 (触发 nz swizzle)

  编译器自动:
    smem 分配、swizzle 参数选择、ldmatrix 插入、
    mma.sync 指令选择、寄存器分配


PyPTO 的 Tiling (3 级):

  Global Memory ──→ L1 SRAM ──→ L0 Buffer ──→ Cube Unit
  (HBM)                                        (矩阵乘法)
        ↑              ↑           ↑
    自动管理       set_cube_       自动管理
                   tile_shapes

  开发者控制:
    set_cube_tile_shapes([m_L0, m_L1], [k_L0, k_L1], [n_L0, n_L1])
    enable_split_k          (K 轴切分)
    pass_options            (编译优化选项)

  框架自动:
    数据搬运指令、L0/L1 缓存调度、Cube 指令选择、
    软件流水线、任务 DAG 构建
${BT3}

### Tile Shape 参数对应关系

| Triton | PyPTO | 含义 |
|--------|-------|------|
| ${BT}BLOCK_M=128${BT} | ${BT}m_L1=128${BT} | M 轴每次处理的行数 |
| ${BT}BLOCK_N=128${BT} | ${BT}n_L1=128${BT} | N 轴每次处理的列数 |
| ${BT}BLOCK_K=32${BT} | ${BT}k_L1=512${BT} | K 轴每次累加的深度 |
| ${BT}num_stages=3${BT} | (自动流水) | 软件流水线深度 |
| ${BT}order=(1,0)${BT} | ${BT}TILEOP_NZ${BT} | 数据布局 / swizzle |

注意 K 轴的差异：Triton 的 BLOCK_K 通常较小（32-64），因为 shared memory 有限；
PyPTO 的 k_L1 可以很大（512+），因为昇腾 SRAM 容量更大。`);

// ============================================================
// Cell 9: 27.6 ND vs NZ cross-platform
// ============================================================
md(`## 27.6 ND vs NZ 格式：跨平台统一概念

有趣的是，ND (朴素布局) 和 NZ (优化布局) 的概念在两个平台上都存在，但含义略有不同。

### 在 NVIDIA GPU / Triton 中

${BT3}
ND = 朴素行主序 shared memory 布局 (有 bank conflict)
NZ = XOR swizzle 后的 shared memory 布局 (无 bank conflict)

地址变换: swizzled_col = col ^ ((row / perPhase) % maxPhase)

作用: 消除 ldmatrix 从 smem 加载时的 bank conflict
控制: order=(1,0) 参数触发编译器自动选择 swizzle
${BT3}

详见 [第17章 §17.7](../03_matmul_optimization/17_matmul_tensorcore.ipynb)

### 在华为昇腾 NPU / PyPTO 中

${BT3}
ND (TILEOP_ND) = 标准 N 维行主序格式
  存储: 数据按行连续存放，最后一维最快变化
  用途: 通用张量操作（向量运算、element-wise）

NZ (TILEOP_NZ) = 分形矩阵格式 (Fractal Z)
  存储: 矩阵被切分为小 tile，tile 间列主序 (N 形)，
        tile 内行主序 (Z 形) → "NZ" 得名
  用途: Cube Unit 矩阵乘法的最优输入格式

  NZ 格式示意 (8x16 FP16 矩阵, tile=16x16):
  ┌──────────────┐
  │ tile(0,0)    │  ← 16x16 的小矩阵块
  │ Z 形排列     │     内部按行存储
  │  ↙↘↙↘↙↘    │
  ├──────────────┤
  │ tile(1,0)    │  ← 列方向下一个 tile
  │  ↙↘↙↘↙↘    │     tile 间按列排列 (N 形)
  └──────────────┘
${BT3}

### 跨平台对比

| 维度 | NVIDIA (Triton) | 昇腾 (PyPTO) |
|------|----------------|-------------|
| ND 含义 | 行主序 smem, 无 swizzle | 标准行主序内存布局 |
| NZ 含义 | XOR swizzle smem 布局 | 分形 Z 矩阵格式 |
| NZ 目的 | 消除 bank conflict | 匹配 Cube Unit 输入格式 |
| 开发者控制 | ${BT}order=(1,0)${BT} 隐式触发 | ${BT}TILEOP_NZ${BT} 显式指定 |
| 格式转换 | 编译器自动 | 框架自动 (或 ${BT}c_matrix_nz=True${BT}) |

**共同点**：两者的 NZ 格式都是为了让数据到达矩阵乘法硬件单元时的访问模式最优化。
在 GPU 上是消除 shared memory bank conflict；在 NPU 上是匹配 Cube Unit 的分形输入格式。`);

// ============================================================
// Cell 10: 27.7 matmul API deep dive
// ============================================================
md(`## 27.7 pypto.matmul API 详解

${BT3}python
pypto.matmul(
    input,              # 左矩阵 A
    mat2,               # 右矩阵 B
    out_dtype,          # 输出数据类型
    *,
    a_trans=False,      # 是否转置 A
    b_trans=False,      # 是否转置 B
    c_matrix_nz=False,  # 输出是否使用 NZ 格式
    extend_params=None, # 扩展参数 (bias, quant, relu)
) -> Tensor
${BT3}

### 支持的数据类型

| 输入类型 | 支持的输出类型 | 说明 |
|---------|-------------|------|
| DT_FP16 | DT_FP16, DT_FP32 | 半精度，最常用 |
| DT_BF16 | DT_BF16, DT_FP32 | Brain Float 16 |
| DT_FP32 | DT_FP32 | 单精度 |
| DT_INT8 | DT_INT32 | 量化推理 |
| DT_FP8E4M3 | DT_FP16, DT_BF16, DT_FP32 | FP8 格式 |
| DT_HF8 | DT_FP16, DT_FP32 | Hybrid FP8 |

### 关键约束

1. **左右矩阵类型必须一致** — 不支持混合精度输入
2. **调用前必须配置 TileShape** — ${BT}pto.set_cube_tile_shapes(...)${BT}
3. **NZ 格式要求 32 字节对齐** — 最内维必须是 16 的倍数 (FP16)
4. **支持 2D/3D/4D 张量** — batch matmul 原生支持

### extend_params 扩展功能

${BT3}python
# 带 bias 的矩阵乘法: C = A @ B + bias
c = pto.matmul(a, b, out_dtype=pto.DT_FP32,
    extend_params={"bias": bias_tensor}
)

# 带 ReLU 激活的矩阵乘法: C = relu(A @ B)
c = pto.matmul(a, b, out_dtype=pto.DT_FP32,
    extend_params={"relu_en": True}
)

# 量化 matmul: C = dequant(A_int8 @ B_int8) * scale
c = pto.matmul(a_int8, b_int8, out_dtype=pto.DT_INT32,
    extend_params={"quant_scale": scale_tensor}
)
${BT3}

对比 Triton 中的算子融合（第10章），PyPTO 通过 ${BT}extend_params${BT} 在一次 matmul 调用中
完成融合，而 Triton 需要在 kernel 中手动添加 bias/activation 代码。`);

// ============================================================
// Cell 11: Triton comparison kernel (runnable)
// ============================================================
code(`# Triton 中的 matmul + bias + relu 融合需要手动编写
@triton.jit
def matmul_bias_relu_kernel(
    a_ptr, b_ptr, bias_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
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

    # 手动融合: + bias + relu
    bias_offs = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    bias = tl.load(bias_ptr + bias_offs, mask=bias_offs < N)
    acc = acc + bias[None, :]        # broadcast bias
    acc = tl.maximum(acc, 0.0)       # relu

    c_block_ptr = tl.make_block_ptr(
        base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
        offsets=(pid_m * BLOCK_M, pid_n * BLOCK_N),
        block_shape=(BLOCK_M, BLOCK_N), order=(1, 0),
    )
    tl.store(c_block_ptr, acc.to(tl.float16), boundary_check=(0, 1))

print("Triton matmul + bias + relu 融合 kernel 定义完成")
print("对比: Triton 需要 ~40 行手动融合 vs PyPTO 的 extend_params={'relu_en': True}")`);

// ============================================================
// Cell 12: 27.8 Performance optimization
// ============================================================
md(`## 27.8 性能优化策略对比

### 算术强度 (Arithmetic Intensity)

两个框架都以算术强度 (AI) 作为优化的核心指标：

$$AI = \\frac{2 \\times M \\times N \\times K}{2 \\times (M \\times K + K \\times N + M \\times N)} \\text{ (FP16, FLOPs/Byte)}$$

当 $AI$ 超过硬件的计算带宽比时，性能从访存瓶颈变为计算瓶颈。

### Triton 优化手段

${BT3}
1. Block Size 调优 (@triton.autotune)
   configs = [
     Config(BLOCK_M=128, BLOCK_N=256, BLOCK_K=32, num_stages=3),
     Config(BLOCK_M=256, BLOCK_N=128, BLOCK_K=32, num_stages=3),
     ...
   ]
   → 自动搜索最优配置

2. L2 Cache Swizzle (grouped_ordering)
   pid = pid_m * grid_n + pid_n  → 改为分组顺序
   → 相邻 program 访问相邻内存，提升 L2 命中率

3. 软件流水线 (num_stages)
   → 预取下一轮数据，隐藏访存延迟

4. Split-K 并行
   → K 维太大时，拆分 K 维让更多 SM 参与
${BT3}

### PyPTO 优化手段

${BT3}
1. Cube Tile Shape 调优
   pto.set_cube_tile_shapes([128, 128], [64, 256], [128, 128])
   → 手动选择最优 tile shape

2. L2 Cache 命中率优化
   通过循环结构控制数据局部性:
   nDim × nL1 = mDim × mL1  → 最大化单次迭代 L2 命中
   → 效果: M=N=K=6144, 从 220 TFLOPS 提升到 290 TFLOPS

3. 自动软件流水线
   框架自动构建 task DAG，实现数据搬运和计算的重叠
   → 不需要开发者手动设置 num_stages

4. K 轴核间切分
   pto.set_cube_tile_shapes(..., enable_split_k=True)
   → 类似 Triton Split-K，但由框架自动管理归约
${BT3}

### 优化策略对比表

| 优化技术 | Triton | PyPTO |
|---------|--------|-------|
| Tile/Block 搜索 | ${BT}@triton.autotune${BT} 自动搜索 | 手动设置 + toolkit 分析 |
| L2 优化 | Swizzle 访问顺序 | 循环结构 + 维度平衡 |
| 流水线 | ${BT}num_stages${BT} 参数 | 自动 (task DAG) |
| Split-K | 手动实现 kernel | ${BT}enable_split_k=True${BT} |
| 算子融合 | 手动写在 kernel 中 | ${BT}extend_params${BT} |
| 性能分析 | NSight, NCU | PyPTO Toolkit 泳道图 |`);

// ============================================================
// Cell 13: 27.9 Compilation pipeline
// ============================================================
md(`## 27.9 编译流水线对比

${BT3}
Triton 编译流水线:
  Python → TTIR → TTGIR → LLVM IR → PTX → CUBIN
    │        │       │       │       │       │
    │     Triton   GPU      LLVM   NVIDIA   最终
    │      AST    特化     优化   编译器   二进制
    │
  tl.dot → mma.sync, ldmatrix, cp.async (自动选择)


PyPTO 编译流水线:
  Python → Tensor Graph → Tile Graph → Block Graph → Exec Graph → PTO 指令 → 硬件二进制
    │          │             │            │            │            │
    │       记录模式       Tile 展开    图分区      调度映射     虚拟指令    NPU 二进制
    │       (tracing)    (按 tile     (子图 =     (MPMD      (类似       (最终
    │                     shape)      kernel)    task DAG)    PTX)       可执行)
    │
  pto.matmul → MAMULB 指令 (Cube Unit 矩阵乘)
${BT3}

### 关键差异

| 阶段 | Triton | PyPTO |
|------|--------|-------|
| 前端 | Python DSL (${BT}tl.*${BT}) | Python API (${BT}pto.*${BT}) |
| 图表示 | TTIR (MLIR 方言) | Tensor/Tile/Block Graph |
| 硬件映射 | TTGIR → LLVM IR | Exec Graph → PTO 虚拟指令 |
| 指令集 | PTX → SASS | PTO-ISA → NPU 二进制 |
| JIT 策略 | 首次调用编译, 缓存 | 首次调用 tracing + 编译, 缓存 |
| 编译产物 | ${BT}.cubin${BT} | NPU 二进制 |`);

// ============================================================
// Cell 14: 27.10 PyTorch integration comparison
// ============================================================
md(`## 27.10 PyTorch 集成对比

两个框架都支持与 PyTorch 无缝集成，但方式不同。

### Triton + PyTorch

${BT3}python
# 方式 1: torch.autograd.Function 包装 (见第 26 章)
class TritonMatmul(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        c = torch.empty(...)
        matmul_kernel[grid](a, b, c, ...)
        return c

# 方式 2: torch.compile 自动生成 Triton kernel
@torch.compile
def my_model(x, w):
    return torch.relu(x @ w.T)  # 自动融合为 Triton kernel
${BT3}

### PyPTO + PyTorch

${BT3}python
import pypto as pto

# 方式 1: Eager Mode (单算子即时执行)
@pto.frontend.jit()
def matmul_kernel(a: pto.Tensor, b: pto.Tensor, c: pto.Tensor):
    pto.set_cube_tile_shapes([128, 128], [256, 512], [128, 128])
    c[:] = pto.matmul(a, b, out_dtype=pto.DT_FP32)

# PyTorch 调用
a_torch = torch.randn(M, K, dtype=torch.float16, device='npu')
b_torch = torch.randn(K, N, dtype=torch.float16, device='npu')
c_torch = torch.empty(M, N, dtype=torch.float32, device='npu')

a_pto = pto.from_torch(a_torch, "A")
b_pto = pto.from_torch(b_torch, "B")
c_pto = pto.from_torch(c_torch, "C")
matmul_kernel(a_pto, b_pto, c_pto)  # JIT 编译 + 执行

# 方式 2: ACLGraph Capture (capture & replay, 减少 host 开销)
g = torch.npu.NPUGraph()
with torch.npu.graph(g):
    y = matmul_kernel(a_pto, b_pto, c_pto)
g.replay()  # 后续调用直接 replay, 无 host 调度开销
${BT3}`);

// ============================================================
// Cell 15: Triton benchmark (runnable)
// ============================================================
code(`# 运行 Triton GEMM 并验证正确性
M, N, K = 2048, 2048, 1024
a = torch.randn(M, K, device='cuda', dtype=torch.float16)
b = torch.randn(K, N, device='cuda', dtype=torch.float16)
c = torch.empty(M, N, device='cuda', dtype=torch.float16)

BLOCK_M, BLOCK_N, BLOCK_K = 128, 128, 32
grid = (M // BLOCK_M, N // BLOCK_N)
matmul_kernel[grid](
    a, b, c, M, N, K,
    a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
)

# 正确性验证
c_ref = torch.matmul(a, b)
max_err = (c.float() - c_ref.float()).abs().max().item()
rel_err = torch.norm(c.float() - c_ref.float()) / torch.norm(c_ref.float())
print(f"Triton GEMM 正确性验证:")
print(f"  最大绝对误差: {max_err:.4f}")
print(f"  相对误差: {rel_err:.6f}")
print(f"  通过: {torch.allclose(c.float(), c_ref.float(), atol=1.0)}")

# 性能测试
ms = triton.testing.do_bench(lambda: matmul_kernel[grid](
    a, b, c, M, N, K,
    a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
    BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
))
tflops = 2.0 * M * N * K / (ms * 1e-3) / 1e12
print(f"\\n性能: {ms:.3f} ms, {tflops:.2f} TFLOPS")`);

// ============================================================
// Cell 16: 27.11 Full comparison table
// ============================================================
md(`## 27.11 全面对比总结

| 维度 | Triton (NVIDIA GPU) | PyPTO (华为昇腾 NPU) |
|------|--------------------|--------------------|
| **厂商** | NVIDIA / OpenAI | 华为 CANN |
| **目标硬件** | NVIDIA GPU (Volta+) | 昇腾 NPU (Atlas A2/A3) |
| **抽象级别** | Block/Tile 级 | Tensor 级 (可降到 Tile) |
| **执行模型** | SPMD | MPMD |
| **矩阵乘法** | ${BT}tl.dot()${BT} → mma.sync | ${BT}pto.matmul()${BT} → MAMULB |
| **数据加载** | ${BT}tl.load()${BT} → cp.async + ldmatrix | 自动 TLOAD |
| **Tiling 控制** | BLOCK_M/N/K (constexpr) | set_cube_tile_shapes (L0/L1 两级) |
| **流水线** | num_stages 参数 | 自动 (task DAG) |
| **算子融合** | 手动写在 kernel 中 | extend_params 参数 |
| **Swizzle/NZ** | 编译器自动 (order 触发) | 显式指定 TILEOP_NZ |
| **自动调优** | @triton.autotune | 手动 + Toolkit 分析 |
| **PyTorch 集成** | autograd.Function / torch.compile | pto.frontend.jit / ACLGraph |
| **调试工具** | NSight, TRITON_INTERPRET | PyPTO Toolkit (三栏联动) |
| **编译路径** | TTIR→TTGIR→LLVM→PTX→CUBIN | TensorGraph→TileGraph→PTO-ISA→二进制 |
| **代码量 (GEMM)** | ~30 行 kernel | ~5 行 (Tensor 级) / ~20 行 (JIT) |
| **学习曲线** | 中等 (需理解 GPU 架构) | 较低 (接近 PyTorch API) |
| **生态成熟度** | 成熟 (大量开源项目) | 早期 (v0.1.2 Beta) |
| **开源协议** | MIT | CANN License 2.0 |

### 各有所长

${BT3}
Triton 的优势:
  ✓ 生态成熟，社区庞大
  ✓ 开发者对底层有更多控制权
  ✓ @triton.autotune 自动搜索最优配置
  ✓ torch.compile 深度集成
  ✓ 支持几乎所有 NVIDIA GPU

PyPTO 的优势:
  ✓ 抽象层级更高，上手更快
  ✓ 算子融合通过参数完成，不需要修改 kernel
  ✓ MPMD 执行模型避免全局同步
  ✓ 自动软件流水线（不需要手动 num_stages）
  ✓ 白盒工具链（Toolkit 三栏联动可视化）
  ✓ 更适合昇腾多核异构架构
${BT3}`);

// ============================================================
// Cell 17: Summary
// ============================================================
md(`## 27.12 本章总结

### 核心收获

1. **编程模型决定一切** — SPMD (Triton) vs MPMD (PyPTO) 不是随意选择，是硬件架构决定的
2. **Tiling 是通用概念** — 无论 GPU 还是 NPU，高性能矩阵乘法都需要将大矩阵分块到片上存储
3. **ND/NZ 格式跨平台存在** — GPU 上的 nz swizzle 和 NPU 上的 TILEOP_NZ 都是为了匹配硬件计算单元的最优访问模式
4. **抽象 vs 控制的权衡** — Triton 给更多底层控制，PyPTO 给更高抽象；两者都在向中间靠拢

### 设计哲学对比

${BT3}
Triton 哲学: "给你一把好用的锤子"
  → 你仍然需要理解 GPU 架构来写出好的 kernel
  → 编译器帮你处理指令选择和寄存器分配
  → 控制权在开发者手中

PyPTO 哲学: "Human-In-The-Loop"
  → 框架先给一个能跑的方案
  → 开发者通过 Toolkit 观察性能瓶颈
  → 通过 pass_options / tile_shapes 迭代优化
  → 工具辅助决策，不试图全自动
${BT3}

### 练习

1. **概念映射**：将第18章终极 GEMM 的 Triton 优化技巧，逐一映射到 PyPTO 中的等价操作
2. **API 对比**：写出 PyPTO 版本的 "matmul + LayerNorm 融合"（解释性代码即可）
3. **思考题**：如果要在同一个推理服务中同时支持 NVIDIA GPU 和昇腾 NPU，架构应该如何设计？

---
← [第26章：生产集成](26_custom_ops.ipynb)`);


// ============================================================
// Build notebook
// ============================================================
const notebook = {
  nbformat: 4, nbformat_minor: 5,
  metadata: {
    kernelspec: { display_name: 'Python 3', language: 'python', name: 'python3' },
    language_info: { name: 'python', version: '3.10.0' }
  },
  cells: cells
};

// Lint check: no mermaid in code cells
cells.forEach((cell, i) => {
  if (cell.cell_type === 'code' && cell.source.join('').includes('```mermaid')) {
    console.warn(`WARNING: Cell ${i} is a code cell but contains mermaid diagram`);
  }
});

const output = JSON.stringify(notebook, null, 1);
fs.writeFileSync('27_pypto_matmul.ipynb', output);
console.log(`Cells: ${cells.length}  Size: ${output.length} bytes`);
