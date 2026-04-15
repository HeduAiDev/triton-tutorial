"""Extract PTX code snippets for each optimization stage."""
import torch, triton, triton.language as tl, re

@triton.jit
def _kern(a_ptr, b_ptr, c_ptr, M, N, K,
    stride_am, stride_ak, stride_bk, stride_bn, stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr):
    pid_m = tl.program_id(0); pid_n = tl.program_id(1)
    a_bp = tl.make_block_ptr(base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak),
        offsets=(pid_m*BLOCK_M, 0), block_shape=(BLOCK_M, BLOCK_K), order=(1, 0))
    b_bp = tl.make_block_ptr(base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn),
        offsets=(0, pid_n*BLOCK_N), block_shape=(BLOCK_K, BLOCK_N), order=(1, 0))
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        acc = tl.dot(tl.load(a_bp, boundary_check=(0,1)), tl.load(b_bp, boundary_check=(0,1)), acc=acc)
        a_bp = tl.advance(a_bp, (0, BLOCK_K)); b_bp = tl.advance(b_bp, (BLOCK_K, 0))
    c_bp = tl.make_block_ptr(base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
        offsets=(pid_m*BLOCK_M, pid_n*BLOCK_N), block_shape=(BLOCK_M, BLOCK_N), order=(1, 0))
    tl.store(c_bp, acc.to(tl.float16), boundary_check=(0,1))

M,N,K = 2048,2048,1024
c1 = _kern.warmup(torch.float16, torch.float16, torch.float16,
    M,N,K, K,1,N,1,N,1, BLOCK_M=128,BLOCK_N=128,BLOCK_K=32, num_stages=1,num_warps=4, grid=(1,))
ptx1 = c1.asm.get('ptx','')
c3 = _kern.warmup(torch.float16, torch.float16, torch.float16,
    M,N,K, K,1,N,1,N,1, BLOCK_M=128,BLOCK_N=128,BLOCK_K=32, num_stages=3,num_warps=4, grid=(1,))
ptx3 = c3.asm.get('ptx','')

lines1 = ptx1.strip().split('\n')
lines3 = ptx3.strip().split('\n')
print(f'stages=1: {len(lines1)} lines')
print(f'stages=3: {len(lines3)} lines')
print()

# stages=1: show ld.global + st.shared + bar.sync (sync data path)
print('=== stages=1: 同步内存拷贝路径 ===')
count = 0
for i, line in enumerate(lines1):
    s = line.strip()
    if any(x in s for x in ['ld.global', 'st.shared', 'bar.sync']) and count < 12:
        print(f'  L{i:>4}: {s}')
        count += 1

print()

# stages=1: show mma.sync
print('=== stages=1: Tensor Core 计算 ===')
count = 0
for i, line in enumerate(lines1):
    s = line.strip()
    if 'mma.sync' in s and count < 4:
        print(f'  L{i:>4}: {s}')
        count += 1

print()

# stages=3: show cp.async and commit/wait
print('=== stages=3: 异步内存拷贝 (cp.async) ===')
count = 0
for i, line in enumerate(lines3):
    s = line.strip()
    if any(x in s for x in ['cp.async']) and count < 8:
        print(f'  L{i:>4}: {s}')
        count += 1

print()
print('=== stages=3: ldmatrix (从 smem 到寄存器) ===')
count = 0
for i, line in enumerate(lines3):
    s = line.strip()
    if 'ldmatrix' in s and count < 4:
        print(f'  L{i:>4}: {s}')
        count += 1

print()
print('=== stages=3: mma.sync (Tensor Core 计算) ===')
count = 0
for i, line in enumerate(lines3):
    s = line.strip()
    if 'mma.sync' in s and count < 4:
        print(f'  L{i:>4}: {s}')
        count += 1

# Save full PTX for reference
with open('/tmp/ptx_stages1.ptx', 'w') as f:
    f.write(ptx1)
with open('/tmp/ptx_stages3.ptx', 'w') as f:
    f.write(ptx3)
print('\nFull PTX saved to /tmp/ptx_stages1.ptx and /tmp/ptx_stages3.ptx')
