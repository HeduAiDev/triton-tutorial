# Part 3 Benchmark Redesign Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make Part 3 GEMM benchmarks fair by using one shared shape set across Ch.12-Ch.18, adding concrete per-chapter numeric tables, and replacing Ch.18's current cumulative TFLOPS chart with one latency-based all-method summary figure.

**Architecture:** Add one small shared benchmark helper module under `03_matmul_optimization/` to centralize shapes, timing, TFLOPS math, and result-table helpers. Keep chapter-specific kernels inside their existing notebooks, then add lightweight table cells that call the shared helper. Ch.18 owns the all-method benchmark harness, the single global latency figure, and the reduced default `ultimate` config space.

**Tech Stack:** Python, PyTorch CUDA events, Triton, pandas, matplotlib, Jupyter notebooks, `unittest` for pure helper tests.

---

## Execution Notes

- Commit steps are included because this plan is written for task-by-task execution. Only run commit commands if the user has explicitly authorized commits in the active session; otherwise leave changes uncommitted and report the changed files.
- Notebook changes should be made with focused cell edits. Do not rewrite entire notebooks unless a cell-level edit becomes impractical.
- Use the existing `conda run -n triton_tutorial ...` environment for notebook execution checks.
- The design spec is `docs/superpowers/specs/2026-04-23-part3-benchmark-redesign.md`.

## File Structure

Create:

- `03_matmul_optimization/benchmark_utils.py`
  - Owns the shared shape set, CUDA timing, TFLOPS math, relative speedup math, correctness helper, DataFrame formatting, and per-shape best summary.
- `tests/test_benchmark_utils.py`
  - Pure unit tests for shape definitions, TFLOPS math, speedup math, and DataFrame conversion. These tests do not require CUDA.

Modify:

- `03_matmul_optimization/12_matmul_smem.ipynb`
  - Add a shared-shape numeric table for `Ch.12 smem` vs `cuBLAS`.
- `03_matmul_optimization/13_matmul_pipeline.ipynb`
  - Add a shared-shape numeric table for `Ch.13 pipeline` vs `Ch.12 smem` vs `cuBLAS`.
- `03_matmul_optimization/14_matmul_splitk.ipynb`
  - Add a shared-shape numeric table for `Ch.14 split-k` vs `Ch.13 pipeline` vs `cuBLAS`.
- `03_matmul_optimization/15_matmul_swizzle.ipynb`
  - Add a shared-shape numeric table for `Ch.15 swizzle` vs `Ch.13 pipeline` vs `cuBLAS`.
- `03_matmul_optimization/16_matmul_fp16_acc.ipynb`
  - Add a shared-shape numeric table for `Ch.16 FP16 accumulator variant` vs `cuBLAS`.
- `03_matmul_optimization/17_matmul_tensorcore.ipynb`
  - Add a shared-shape numeric table for `Ch.17 tensorcore` vs `cuBLAS`.
- `03_matmul_optimization/18_matmul_ultimate.ipynb`
  - Reduce default autotune config count.
  - Replace the current cumulative TFLOPS visualization with one all-method latency result table and one grouped latency figure.
  - Save the figure to `03_matmul_optimization/cumulative_optimization_latency.png`.
- `README.md`
  - Update Part 3 benchmark wording from a single-shape performance table to the shared-shape protocol.

---

### Task 1: Add Shared Benchmark Helper

**Files:**
- Create: `03_matmul_optimization/benchmark_utils.py`
- Create: `tests/test_benchmark_utils.py`

- [ ] **Step 1: Write the helper tests**

Create `tests/test_benchmark_utils.py` with this content:

```python
import math
import unittest

import pandas as pd

from 03_matmul_optimization.benchmark_utils import (
    BENCHMARK_SHAPES,
    BenchmarkResult,
    format_results,
    speedup,
    summarize_best_by_shape,
    tflops_from_ms,
)
```

This import is invalid because Python module names cannot start with digits. Use the load-by-path version instead:

```python
import importlib.util
import math
import pathlib
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "03_matmul_optimization" / "benchmark_utils.py"
spec = importlib.util.spec_from_file_location("benchmark_utils", MODULE_PATH)
benchmark_utils = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(benchmark_utils)


class BenchmarkUtilsTest(unittest.TestCase):
    def test_shape_set_is_fixed_and_categorized(self):
        shapes = benchmark_utils.BENCHMARK_SHAPES
        self.assertEqual(len(shapes), 7)
        self.assertEqual([s.name for s in shapes], [
            "square-2k",
            "square-4k",
            "tall-8k-x-512",
            "tall-16k-x-256",
            "wide-512-x-8k",
            "largeK-2k-x-8k",
            "largeK-1k-x-16k",
        ])
        self.assertEqual({s.category for s in shapes}, {
            "square",
            "tall-skinny",
            "wide-short",
            "large-K",
        })

    def test_tflops_from_ms(self):
        value = benchmark_utils.tflops_from_ms(2048, 2048, 2048, 0.1)
        expected = (2.0 * 2048 * 2048 * 2048) / (0.1 * 1e-3) / 1e12
        self.assertTrue(math.isclose(value, expected, rel_tol=1e-12))

    def test_speedup_uses_latency_ratio(self):
        self.assertEqual(benchmark_utils.speedup(2.0, 1.0), 2.0)
        self.assertEqual(benchmark_utils.speedup(1.0, 2.0), 0.5)
        self.assertTrue(math.isnan(benchmark_utils.speedup(float("nan"), 2.0)))
        self.assertTrue(math.isnan(benchmark_utils.speedup(2.0, 0.0)))

    def test_format_results_builds_dataframe(self):
        rows = [
            benchmark_utils.BenchmarkResult(
                shape_name="square-2k",
                category="square",
                M=2048,
                N=2048,
                K=2048,
                method="cuBLAS",
                latency_ms=0.1,
                tflops=171.8,
            ),
            benchmark_utils.BenchmarkResult(
                shape_name="square-2k",
                category="square",
                M=2048,
                N=2048,
                K=2048,
                method="Ch.18 ultimate",
                latency_ms=0.08,
                tflops=214.7,
            ),
        ]
        df = benchmark_utils.format_results(rows)
        self.assertEqual(list(df.columns), [
            "shape_name",
            "category",
            "M",
            "N",
            "K",
            "method",
            "latency_ms",
            "tflops",
            "speedup_vs_cublas",
            "speedup_vs_previous",
            "max_err",
            "passed",
        ])
        self.assertEqual(len(df), 2)

    def test_summarize_best_by_shape(self):
        rows = [
            benchmark_utils.BenchmarkResult("shape", "square", 1, 1, 1, "cuBLAS", 2.0, 1.0),
            benchmark_utils.BenchmarkResult("shape", "square", 1, 1, 1, "A", 1.0, 2.0),
            benchmark_utils.BenchmarkResult("shape", "square", 1, 1, 1, "B", 3.0, 0.7),
        ]
        summary = benchmark_utils.summarize_best_by_shape(benchmark_utils.format_results(rows))
        self.assertEqual(summary.loc[0, "shape_name"], "shape")
        self.assertEqual(summary.loc[0, "best_method"], "A")
        self.assertEqual(summary.loc[0, "best_triton_ms"], 1.0)
        self.assertEqual(summary.loc[0, "cublas_ms"], 2.0)
        self.assertEqual(summary.loc[0, "best_triton_vs_cublas"], 2.0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the tests and confirm they fail before implementation**

Run:

```bash
python -m unittest tests/test_benchmark_utils.py -v
```

Expected: failure because `03_matmul_optimization/benchmark_utils.py` does not exist.

- [ ] **Step 3: Create the helper module**

Create `03_matmul_optimization/benchmark_utils.py` with this content:

```python
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Callable, Iterable

import pandas as pd
import torch


@dataclass(frozen=True)
class BenchmarkShape:
    name: str
    category: str
    M: int
    N: int
    K: int


@dataclass(frozen=True)
class BenchmarkResult:
    shape_name: str
    category: str
    M: int
    N: int
    K: int
    method: str
    latency_ms: float
    tflops: float
    speedup_vs_cublas: float = math.nan
    speedup_vs_previous: float = math.nan
    max_err: float = math.nan
    passed: bool = True


BENCHMARK_SHAPES = [
    BenchmarkShape("square-2k", "square", 2048, 2048, 2048),
    BenchmarkShape("square-4k", "square", 4096, 4096, 4096),
    BenchmarkShape("tall-8k-x-512", "tall-skinny", 8192, 512, 2048),
    BenchmarkShape("tall-16k-x-256", "tall-skinny", 16384, 256, 2048),
    BenchmarkShape("wide-512-x-8k", "wide-short", 512, 8192, 2048),
    BenchmarkShape("largeK-2k-x-8k", "large-K", 2048, 2048, 8192),
    BenchmarkShape("largeK-1k-x-16k", "large-K", 1024, 1024, 16384),
]

DEFAULT_WARMUP = 5
DEFAULT_REP = 20


def tflops_from_ms(M: int, N: int, K: int, latency_ms: float) -> float:
    if latency_ms <= 0 or math.isnan(latency_ms):
        return math.nan
    return (2.0 * M * N * K) / (latency_ms * 1e-3) / 1e12


def speedup(reference_ms: float, candidate_ms: float) -> float:
    if reference_ms <= 0 or candidate_ms <= 0:
        return math.nan
    if math.isnan(reference_ms) or math.isnan(candidate_ms):
        return math.nan
    return reference_ms / candidate_ms


def make_fp16_inputs(M: int, N: int, K: int, seed: int = 42) -> tuple[torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    return a, b


def time_cuda(fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
              a: torch.Tensor,
              b: torch.Tensor,
              warmup: int = DEFAULT_WARMUP,
              rep: int = DEFAULT_REP) -> float:
    for _ in range(warmup):
        fn(a, b)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(rep):
        fn(a, b)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / rep


def max_abs_error(candidate: torch.Tensor, reference: torch.Tensor) -> float:
    return (candidate - reference).abs().max().item()


def benchmark_method(method_name: str,
                     fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                     shape: BenchmarkShape,
                     a: torch.Tensor,
                     b: torch.Tensor,
                     c_ref: torch.Tensor | None = None,
                     warmup: int = DEFAULT_WARMUP,
                     rep: int = DEFAULT_REP,
                     atol: float = 2.0) -> BenchmarkResult:
    try:
        candidate = fn(a, b)
        max_err = max_abs_error(candidate, c_ref) if c_ref is not None else math.nan
        passed = True if c_ref is None else max_err <= atol
        latency_ms = time_cuda(fn, a, b, warmup=warmup, rep=rep)
        tflops = tflops_from_ms(shape.M, shape.N, shape.K, latency_ms)
        return BenchmarkResult(
            shape_name=shape.name,
            category=shape.category,
            M=shape.M,
            N=shape.N,
            K=shape.K,
            method=method_name,
            latency_ms=latency_ms,
            tflops=tflops,
            max_err=max_err,
            passed=passed,
        )
    except Exception:
        return BenchmarkResult(
            shape_name=shape.name,
            category=shape.category,
            M=shape.M,
            N=shape.N,
            K=shape.K,
            method=method_name,
            latency_ms=math.nan,
            tflops=math.nan,
            max_err=math.nan,
            passed=False,
        )


def format_results(results: Iterable[BenchmarkResult]) -> pd.DataFrame:
    columns = [
        "shape_name",
        "category",
        "M",
        "N",
        "K",
        "method",
        "latency_ms",
        "tflops",
        "speedup_vs_cublas",
        "speedup_vs_previous",
        "max_err",
        "passed",
    ]
    rows = [asdict(result) for result in results]
    return pd.DataFrame(rows, columns=columns)


def add_relative_columns(df: pd.DataFrame,
                         cublas_method: str = "cuBLAS",
                         previous_method_by_method: dict[str, str] | None = None) -> pd.DataFrame:
    out = df.copy()
    previous_method_by_method = previous_method_by_method or {}

    for shape_name in out["shape_name"].drop_duplicates():
        shape_mask = out["shape_name"] == shape_name
        cublas_rows = out.loc[shape_mask & (out["method"] == cublas_method), "latency_ms"]
        cublas_ms = float(cublas_rows.iloc[0]) if len(cublas_rows) else math.nan
        for idx in out.loc[shape_mask].index:
            out.at[idx, "speedup_vs_cublas"] = speedup(cublas_ms, float(out.at[idx, "latency_ms"]))
            prev_method = previous_method_by_method.get(str(out.at[idx, "method"]))
            if prev_method:
                prev_rows = out.loc[shape_mask & (out["method"] == prev_method), "latency_ms"]
                prev_ms = float(prev_rows.iloc[0]) if len(prev_rows) else math.nan
                out.at[idx, "speedup_vs_previous"] = speedup(prev_ms, float(out.at[idx, "latency_ms"]))
    return out


def summarize_best_by_shape(df: pd.DataFrame, cublas_method: str = "cuBLAS") -> pd.DataFrame:
    rows = []
    for shape_name, group in df.groupby("shape_name", sort=False):
        valid = group[group["latency_ms"].notna()].copy()
        cublas = valid[valid["method"] == cublas_method]
        triton = valid[valid["method"] != cublas_method]
        if triton.empty:
            continue
        best = triton.sort_values("latency_ms", ascending=True).iloc[0]
        cublas_ms = float(cublas.iloc[0]["latency_ms"]) if not cublas.empty else math.nan
        rows.append({
            "shape_name": shape_name,
            "category": best["category"],
            "best_method": best["method"],
            "best_triton_ms": float(best["latency_ms"]),
            "cublas_ms": cublas_ms,
            "best_triton_vs_cublas": speedup(cublas_ms, float(best["latency_ms"])),
        })
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run helper tests and confirm they pass**

Run:

```bash
python -m unittest tests/test_benchmark_utils.py -v
```

Expected: `Ran 5 tests` and `OK`.

- [ ] **Step 5: Commit helper changes if commits are authorized**

```bash
git add 03_matmul_optimization/benchmark_utils.py tests/test_benchmark_utils.py
git commit -m "Add shared Part 3 benchmark utilities"
```

Expected: one new commit containing only the helper and its tests.

---

### Task 2: Add Ch.12 Shared-Shape Numeric Table

**Files:**
- Modify: `03_matmul_optimization/12_matmul_smem.ipynb`

- [ ] **Step 1: Add a markdown cell before the existing Ch.12 summary section**

Insert this markdown cell near the end of the notebook, after the existing cuBLAS comparison cell:

```markdown
## 12.x 统一 Shape Set 数值表

为了让 Part 3 的所有优化章节可横向比较，本表使用 Ch.18 中统一定义的 7 个矩阵形状。

本章只比较 `Ch.12 smem` 与 `cuBLAS`，主指标是 `latency_ms`，并附带 `TFLOPS` 和相对 cuBLAS 的速度比。
```

- [ ] **Step 2: Add the Ch.12 numeric table code cell**

Insert this code cell after the markdown cell:

```python
import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))
from benchmark_utils import (
    BENCHMARK_SHAPES,
    add_relative_columns,
    benchmark_method,
    format_results,
    make_fp16_inputs,
)

print("Ch.12 shared-shape benchmark: smem vs cuBLAS")
chapter12_results = []

for shape in BENCHMARK_SHAPES:
    a, b = make_fp16_inputs(shape.M, shape.N, shape.K)
    c_ref = torch.matmul(a, b)
    methods = {
        "Ch.12 smem": matmul_block_ptr,
        "cuBLAS": lambda x, y: torch.matmul(x, y),
    }
    for method_name, fn in methods.items():
        chapter12_results.append(
            benchmark_method(method_name, fn, shape, a, b, c_ref=c_ref)
        )
    del a, b, c_ref
    torch.cuda.empty_cache()

chapter12_df = add_relative_columns(format_results(chapter12_results))
chapter12_df[[
    "shape_name", "category", "method", "latency_ms", "tflops", "speedup_vs_cublas", "max_err", "passed"
]]
```

- [ ] **Step 3: Execute Ch.12 smoke test**

Run:

```bash
conda run -n triton_tutorial jupyter nbconvert --to notebook --execute 03_matmul_optimization/12_matmul_smem.ipynb --output /tmp/ch12_benchmark_check.ipynb
```

Expected: command exits with code 0 and the final notebook contains a table named `chapter12_df`.

- [ ] **Step 4: Commit Ch.12 changes if commits are authorized**

```bash
git add 03_matmul_optimization/12_matmul_smem.ipynb
git commit -m "Add Ch12 shared-shape benchmark table"
```

Expected: one commit modifying only Ch.12.

---

### Task 3: Add Ch.13 Shared-Shape Numeric Table

**Files:**
- Modify: `03_matmul_optimization/13_matmul_pipeline.ipynb`

- [ ] **Step 1: Add a markdown cell near the end of Ch.13**

```markdown
## 13.x 统一 Shape Set 数值表

本表使用 Part 3 的统一 7 个矩阵形状，比较 `Ch.13 pipeline`、`Ch.12 smem` 与 `cuBLAS`。

`vs previous chapter` 表示本章 pipeline 相对于 Ch.12 smem 的延迟速度比。
```

- [ ] **Step 2: Add the Ch.13 numeric table code cell**

```python
import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))
from benchmark_utils import (
    BENCHMARK_SHAPES,
    add_relative_columns,
    benchmark_method,
    format_results,
    make_fp16_inputs,
)

print("Ch.13 shared-shape benchmark: pipeline vs smem vs cuBLAS")
chapter13_results = []
previous_map = {"Ch.13 pipeline": "Ch.12 smem"}

for shape in BENCHMARK_SHAPES:
    a, b = make_fp16_inputs(shape.M, shape.N, shape.K)
    c_ref = torch.matmul(a, b)
    methods = {
        "Ch.12 smem": lambda x, y: matmul_pipeline(x, y, num_stages=1),
        "Ch.13 pipeline": lambda x, y: matmul_pipeline(x, y, num_stages=3),
        "cuBLAS": lambda x, y: torch.matmul(x, y),
    }
    for method_name, fn in methods.items():
        chapter13_results.append(
            benchmark_method(method_name, fn, shape, a, b, c_ref=c_ref)
        )
    del a, b, c_ref
    torch.cuda.empty_cache()

chapter13_df = add_relative_columns(
    format_results(chapter13_results),
    previous_method_by_method=previous_map,
)
chapter13_df[[
    "shape_name", "category", "method", "latency_ms", "tflops",
    "speedup_vs_cublas", "speedup_vs_previous", "max_err", "passed"
]]
```

- [ ] **Step 3: Execute Ch.13 smoke test**

Run:

```bash
conda run -n triton_tutorial jupyter nbconvert --to notebook --execute 03_matmul_optimization/13_matmul_pipeline.ipynb --output /tmp/ch13_benchmark_check.ipynb
```

Expected: command exits with code 0 and the final notebook contains `chapter13_df`.

- [ ] **Step 4: Commit Ch.13 changes if commits are authorized**

```bash
git add 03_matmul_optimization/13_matmul_pipeline.ipynb
git commit -m "Add Ch13 shared-shape benchmark table"
```

Expected: one commit modifying only Ch.13.

---

### Task 4: Add Ch.14 Shared-Shape Numeric Table

**Files:**
- Modify: `03_matmul_optimization/14_matmul_splitk.ipynb`

- [ ] **Step 1: Add a markdown cell near the end of Ch.14**

```markdown
## 14.x 统一 Shape Set 数值表

本表使用 Part 3 的统一 7 个矩阵形状，比较 `Ch.14 split-k`、`Ch.13 pipeline` 与 `cuBLAS`。

Split-K 是 shape-sensitive 优化：它在 tall-skinny 或 program 数不足时可能有收益，在方阵上变慢也属于预期现象。
```

- [ ] **Step 2: Add the Ch.14 numeric table code cell**

```python
import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))
from benchmark_utils import (
    BENCHMARK_SHAPES,
    add_relative_columns,
    benchmark_method,
    format_results,
    make_fp16_inputs,
)

print("Ch.14 shared-shape benchmark: split-k vs pipeline vs cuBLAS")
chapter14_results = []
previous_map = {"Ch.14 split-k": "Ch.13 pipeline"}

for shape in BENCHMARK_SHAPES:
    a, b = make_fp16_inputs(shape.M, shape.N, shape.K)
    c_ref = torch.matmul(a, b)
    methods = {
        "Ch.13 pipeline": lambda x, y: matmul_standard(x, y, num_stages=3),
        "Ch.14 split-k": lambda x, y: matmul_splitk(x, y, SPLIT_K=4, num_stages=3),
        "cuBLAS": lambda x, y: torch.matmul(x, y),
    }
    for method_name, fn in methods.items():
        chapter14_results.append(
            benchmark_method(method_name, fn, shape, a, b, c_ref=c_ref)
        )
    del a, b, c_ref
    torch.cuda.empty_cache()

chapter14_df = add_relative_columns(
    format_results(chapter14_results),
    previous_method_by_method=previous_map,
)
chapter14_df[[
    "shape_name", "category", "method", "latency_ms", "tflops",
    "speedup_vs_cublas", "speedup_vs_previous", "max_err", "passed"
]]
```

- [ ] **Step 3: Execute Ch.14 smoke test**

Run:

```bash
conda run -n triton_tutorial jupyter nbconvert --to notebook --execute 03_matmul_optimization/14_matmul_splitk.ipynb --output /tmp/ch14_benchmark_check.ipynb
```

Expected: command exits with code 0 and the final notebook contains `chapter14_df`. Some split-k rows may be slower than pipeline.

- [ ] **Step 4: Commit Ch.14 changes if commits are authorized**

```bash
git add 03_matmul_optimization/14_matmul_splitk.ipynb
git commit -m "Add Ch14 shared-shape benchmark table"
```

Expected: one commit modifying only Ch.14.

---

### Task 5: Add Ch.15 Shared-Shape Numeric Table

**Files:**
- Modify: `03_matmul_optimization/15_matmul_swizzle.ipynb`

- [ ] **Step 1: Add a markdown cell near the end of Ch.15**

```markdown
## 15.x 统一 Shape Set 数值表

本表使用 Part 3 的统一 7 个矩阵形状，比较 `Ch.15 swizzle`、`Ch.13 pipeline` 与 `cuBLAS`。

Swizzle 优化的是 block 调度和 L2 locality，不改变 `tl.dot` 的 Tensor Core 计算路径。
```

- [ ] **Step 2: Add the Ch.15 numeric table code cell**

```python
import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))
from benchmark_utils import (
    BENCHMARK_SHAPES,
    add_relative_columns,
    benchmark_method,
    format_results,
    make_fp16_inputs,
)

print("Ch.15 shared-shape benchmark: swizzle vs pipeline vs cuBLAS")
chapter15_results = []
previous_map = {"Ch.15 swizzle": "Ch.13 pipeline"}

for shape in BENCHMARK_SHAPES:
    a, b = make_fp16_inputs(shape.M, shape.N, shape.K)
    c_ref = torch.matmul(a, b)
    methods = {
        "Ch.13 pipeline": lambda x, y: matmul_pipeline(x, y, num_stages=3),
        "Ch.15 swizzle": lambda x, y: matmul_swizzle(x, y, GROUP_SIZE_M=8, num_stages=3),
        "cuBLAS": lambda x, y: torch.matmul(x, y),
    }
    for method_name, fn in methods.items():
        chapter15_results.append(
            benchmark_method(method_name, fn, shape, a, b, c_ref=c_ref)
        )
    del a, b, c_ref
    torch.cuda.empty_cache()

chapter15_df = add_relative_columns(
    format_results(chapter15_results),
    previous_method_by_method=previous_map,
)
chapter15_df[[
    "shape_name", "category", "method", "latency_ms", "tflops",
    "speedup_vs_cublas", "speedup_vs_previous", "max_err", "passed"
]]
```

- [ ] **Step 3: Execute Ch.15 smoke test**

Run:

```bash
conda run -n triton_tutorial jupyter nbconvert --to notebook --execute 03_matmul_optimization/15_matmul_swizzle.ipynb --output /tmp/ch15_benchmark_check.ipynb
```

Expected: command exits with code 0 and the final notebook contains `chapter15_df`.

- [ ] **Step 4: Commit Ch.15 changes if commits are authorized**

```bash
git add 03_matmul_optimization/15_matmul_swizzle.ipynb
git commit -m "Add Ch15 shared-shape benchmark table"
```

Expected: one commit modifying only Ch.15.

---

### Task 6: Add Ch.16 Shared-Shape Numeric Table

**Files:**
- Modify: `03_matmul_optimization/16_matmul_fp16_acc.ipynb`

- [ ] **Step 1: Add a markdown cell near the end of Ch.16**

```markdown
## 16.x 统一 Shape Set 数值表

本表使用 Part 3 的统一 7 个矩阵形状，比较 `Ch.16 FP16 input + FP32 accumulation` 与 `cuBLAS`。

本章关注的是精度策略，因此表格保留 `max_err` 和 `passed`，帮助读者同时检查性能和数值正确性。
```

- [ ] **Step 2: Add the Ch.16 numeric table code cell**

```python
import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))
from benchmark_utils import (
    BENCHMARK_SHAPES,
    add_relative_columns,
    benchmark_method,
    format_results,
    make_fp16_inputs,
)

print("Ch.16 shared-shape benchmark: FP16 input + FP32 accumulation vs cuBLAS")
chapter16_results = []

for shape in BENCHMARK_SHAPES:
    a, b = make_fp16_inputs(shape.M, shape.N, shape.K)
    c_ref = torch.matmul(a, b)
    methods = {
        "Ch.16 fp32-acc": lambda x, y: matmul_precision(x, y, allow_tf32=False, output_fp32=False),
        "cuBLAS": lambda x, y: torch.matmul(x, y),
    }
    for method_name, fn in methods.items():
        chapter16_results.append(
            benchmark_method(method_name, fn, shape, a, b, c_ref=c_ref)
        )
    del a, b, c_ref
    torch.cuda.empty_cache()

chapter16_df = add_relative_columns(format_results(chapter16_results))
chapter16_df[[
    "shape_name", "category", "method", "latency_ms", "tflops", "speedup_vs_cublas", "max_err", "passed"
]]
```

- [ ] **Step 3: Execute Ch.16 smoke test**

Run:

```bash
conda run -n triton_tutorial jupyter nbconvert --to notebook --execute 03_matmul_optimization/16_matmul_fp16_acc.ipynb --output /tmp/ch16_benchmark_check.ipynb
```

Expected: command exits with code 0 and the final notebook contains `chapter16_df`.

- [ ] **Step 4: Commit Ch.16 changes if commits are authorized**

```bash
git add 03_matmul_optimization/16_matmul_fp16_acc.ipynb
git commit -m "Add Ch16 shared-shape benchmark table"
```

Expected: one commit modifying only Ch.16.

---

### Task 7: Add Ch.17 Shared-Shape Numeric Table

**Files:**
- Modify: `03_matmul_optimization/17_matmul_tensorcore.ipynb`

- [ ] **Step 1: Add a markdown cell near the end of Ch.17**

```markdown
## 17.x 统一 Shape Set 数值表

本表使用 Part 3 的统一 7 个矩阵形状，比较 `Ch.17 tensorcore` 与 `cuBLAS`。

本章的重点是 `tl.dot` 到 Tensor Core / `mma.sync` 的映射，表格展示同一批形状下 Tensor Core 路径的延迟和吞吐。
```

- [ ] **Step 2: Add the Ch.17 numeric table code cell**

```python
import sys
from pathlib import Path

sys.path.append(str(Path.cwd()))
from benchmark_utils import (
    BENCHMARK_SHAPES,
    add_relative_columns,
    benchmark_method,
    format_results,
    make_fp16_inputs,
)

print("Ch.17 shared-shape benchmark: tensorcore vs cuBLAS")
chapter17_results = []

for shape in BENCHMARK_SHAPES:
    a, b = make_fp16_inputs(shape.M, shape.N, shape.K)
    c_ref = torch.matmul(a, b)
    methods = {
        "Ch.17 tensorcore": lambda x, y: matmul_tc(x, y, BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, GROUP_SIZE_M=8),
        "cuBLAS": lambda x, y: torch.matmul(x, y),
    }
    for method_name, fn in methods.items():
        chapter17_results.append(
            benchmark_method(method_name, fn, shape, a, b, c_ref=c_ref)
        )
    del a, b, c_ref
    torch.cuda.empty_cache()

chapter17_df = add_relative_columns(format_results(chapter17_results))
chapter17_df[[
    "shape_name", "category", "method", "latency_ms", "tflops", "speedup_vs_cublas", "max_err", "passed"
]]
```

- [ ] **Step 3: Execute Ch.17 smoke test**

Run:

```bash
conda run -n triton_tutorial jupyter nbconvert --to notebook --execute 03_matmul_optimization/17_matmul_tensorcore.ipynb --output /tmp/ch17_benchmark_check.ipynb
```

Expected: command exits with code 0 and the final notebook contains `chapter17_df`.

- [ ] **Step 4: Commit Ch.17 changes if commits are authorized**

```bash
git add 03_matmul_optimization/17_matmul_tensorcore.ipynb
git commit -m "Add Ch17 shared-shape benchmark table"
```

Expected: one commit modifying only Ch.17.

---

### Task 8: Reduce Ch.18 Default Autotune And Benchmark Cost

**Files:**
- Modify: `03_matmul_optimization/18_matmul_ultimate.ipynb`

- [ ] **Step 1: Replace the large config generator cell text**

Modify the `get_autotune_configs()` cell so the default config space is small and explicit:

```python
# ========== 教程默认配置空间 ==========
# 默认路径只保留少量代表配置，避免 notebook 执行时做大规模 autotune。
# 更大的搜索空间适合离线调优，不适合作为教程默认执行路径。

def get_autotune_configs():
    """Return a compact teaching config set for the default notebook path."""
    return [
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
    ]

print(f"默认配置空间大小: {len(get_autotune_configs())} 种配置")
print("说明: 大规模 autotune 已从默认路径移除，避免 Ch.18 执行时间接近 10 分钟。")
```

- [ ] **Step 2: Replace the `matmul_fast_kernel` compact config list**

In the existing Ch.18 compact benchmark kernel cell, replace the 13-config list with this 4-config list:

```python
configs=[
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
    triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
    triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=8),
    triton.Config({'BLOCK_M': 64,  'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_SIZE_M': 8}, num_stages=3, num_warps=4),
],
```

- [ ] **Step 3: Reduce default benchmark repetitions in Ch.18**

Change Ch.18 benchmark defaults from `num_warmup=25, num_rep=100` to:

```python
def benchmark_all(M, N, K, num_warmup=5, num_rep=20):
```

Change the shared benchmark helper call sites in Ch.18 to rely on `DEFAULT_WARMUP = 5` and `DEFAULT_REP = 20` from `benchmark_utils.py`.

- [ ] **Step 4: Execute a runtime smoke check for Ch.18**

Run:

```bash
time conda run -n triton_tutorial jupyter nbconvert --to notebook --execute 03_matmul_optimization/18_matmul_ultimate.ipynb --output /tmp/ch18_runtime_check.ipynb
```

Expected: command exits with code 0. Runtime should be clearly below the current about-10-minute experience on RTX PRO 6000.

- [ ] **Step 5: Commit runtime reduction if commits are authorized**

```bash
git add 03_matmul_optimization/18_matmul_ultimate.ipynb
git commit -m "Reduce Ch18 default benchmark runtime"
```

Expected: one commit modifying only Ch.18.

---

### Task 9: Replace Ch.18 Cumulative TFLOPS Visualization With Global Latency Summary

**Files:**
- Modify: `03_matmul_optimization/18_matmul_ultimate.ipynb`
- Create: `03_matmul_optimization/cumulative_optimization_latency.png`
- Modify or remove from tracking if replaced: `03_matmul_optimization/cumulative_optimization.png`

- [ ] **Step 1: Replace the Ch.18 cumulative visualization markdown cell**

Replace the markdown cell headed `## 18.5 累进优化可视化` with:

```markdown
## 18.5 统一 Shape Set 的全方法延迟对比

本节使用 Part 3 统一的 7 个矩阵形状，对 Ch.12~Ch.18 的代表 kernel 和 cuBLAS 做同一协议下的横向比较。

- 主指标：`latency_ms`，越低越好
- 辅助指标：`TFLOPS`、`vs cuBLAS`、最快方法摘要
- 图表职责：只提供一张全局趋势图
- 具体数值：由本节总表和各章节表格提供

注意：Split-K 是 shape-sensitive 优化，在部分 shape 上变慢是预期现象；这些结果会如实保留。
```

- [ ] **Step 2: Replace the old Ch.18 visualization code cell**

Replace the current TFLOPS two-subplot code cell with this code:

```python
# ========== 统一 Shape Set 的全方法延迟对比 ==========
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib
import numpy as np

sys.path.append(str(Path.cwd()))
from benchmark_utils import (
    BENCHMARK_SHAPES,
    add_relative_columns,
    benchmark_method,
    format_results,
    make_fp16_inputs,
    summarize_best_by_shape,
)

matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']

# ---------- 内联定义各阶段 kernel ----------
# 保留现有 _smem_kernel、_pipeline_kernel、_swizzle_kernel、run_smem、run_pipeline、run_swizzle。
# 在实现时只删除旧 shapes/data/TFLOPS plotting 部分，保留这些 kernel wrapper。

# ---------- Split-K wrapper ----------
def run_splitk(a, b):
    return matmul_splitk(a, b, SPLIT_K=4, num_stages=3)

# ---------- Ch.16 / Ch.17 wrappers ----------
def run_precision(a, b):
    return matmul_precision(a, b, allow_tf32=False, output_fp32=False)


def run_tensorcore(a, b):
    return matmul_tc(a, b, BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, GROUP_SIZE_M=8)


methods = {
    "Ch.12 smem": run_smem,
    "Ch.13 pipeline": run_pipeline,
    "Ch.14 split-k": run_splitk,
    "Ch.15 swizzle": run_swizzle,
    "Ch.16 fp32-acc": run_precision,
    "Ch.17 tensorcore": run_tensorcore,
    "Ch.18 ultimate": matmul_fast,
    "cuBLAS": lambda a, b: torch.matmul(a, b),
}

previous_map = {
    "Ch.13 pipeline": "Ch.12 smem",
    "Ch.14 split-k": "Ch.13 pipeline",
    "Ch.15 swizzle": "Ch.13 pipeline",
    "Ch.18 ultimate": "Ch.15 swizzle",
}

all_method_results = []

for shape in BENCHMARK_SHAPES:
    print(f"Benchmarking {shape.name}: M={shape.M}, N={shape.N}, K={shape.K}")
    a, b = make_fp16_inputs(shape.M, shape.N, shape.K)
    c_ref = torch.matmul(a, b)
    for method_name, fn in methods.items():
        result = benchmark_method(method_name, fn, shape, a, b, c_ref=c_ref)
        all_method_results.append(result)
        print(f"  {method_name:>18}: {result.latency_ms:>8.4f} ms, passed={result.passed}")
    del a, b, c_ref
    torch.cuda.empty_cache()

all_methods_df = add_relative_columns(
    format_results(all_method_results),
    previous_method_by_method=previous_map,
)

summary_df = summarize_best_by_shape(all_methods_df)

display(all_methods_df[[
    "shape_name", "category", "method", "latency_ms", "tflops",
    "speedup_vs_cublas", "speedup_vs_previous", "max_err", "passed"
]])

display(summary_df)

# ---------- 单张全局 latency 图 ----------
shape_labels = [shape.name for shape in BENCHMARK_SHAPES]
method_names = list(methods.keys())
x = np.arange(len(shape_labels))
width = 0.10
colors = plt.cm.tab10(np.linspace(0, 1, len(method_names)))

fig, ax = plt.subplots(figsize=(18, 7))
for i, method_name in enumerate(method_names):
    vals = []
    for shape_name in shape_labels:
        rows = all_methods_df[
            (all_methods_df["shape_name"] == shape_name)
            & (all_methods_df["method"] == method_name)
        ]
        vals.append(float(rows.iloc[0]["latency_ms"]) if len(rows) else np.nan)
    offset = (i - len(method_names) / 2 + 0.5) * width
    ax.bar(x + offset, vals, width, label=method_name, color=colors[i])

ax.set_xlabel("Matrix shape (MxNxK)", fontsize=12)
ax.set_ylabel("Latency (ms, lower is better)", fontsize=12)
ax.set_title("Part 3 GEMM Optimization: Shared Shape Latency Comparison", fontsize=14)
ax.set_xticks(x)
ax.set_xticklabels(shape_labels, rotation=25, ha="right", fontsize=9)
ax.legend(fontsize=9, ncols=2)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig("cumulative_optimization_latency.png", dpi=150, bbox_inches="tight")
plt.show()
print("Figure saved to cumulative_optimization_latency.png")
```

During implementation, the code above must be integrated with existing Ch.18 local definitions. If `matmul_splitk`, `matmul_precision`, or `matmul_tc` are not defined in Ch.18, add minimal representative definitions by copying the final wrapper/kernel from Ch.14, Ch.16, and Ch.17 into the Ch.18 benchmark section before this cell.

- [ ] **Step 3: Execute Ch.18 after visualization replacement**

Run:

```bash
conda run -n triton_tutorial jupyter nbconvert --to notebook --execute 03_matmul_optimization/18_matmul_ultimate.ipynb --output /tmp/ch18_latency_summary_check.ipynb
```

Expected: command exits with code 0, creates `03_matmul_optimization/cumulative_optimization_latency.png`, and the executed notebook contains `all_methods_df` plus `summary_df`.

- [ ] **Step 4: Commit Ch.18 summary changes if commits are authorized**

```bash
git add 03_matmul_optimization/18_matmul_ultimate.ipynb 03_matmul_optimization/cumulative_optimization_latency.png
git commit -m "Add Ch18 shared-shape latency summary"
```

Expected: one commit containing the Ch.18 benchmark summary and the generated latency figure.

---

### Task 10: Update README Benchmark Description

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Replace the current single-shape performance target section**

Replace the section starting at `## 性能目标` with:

```markdown
## 性能评测协议

Part 3 的 GEMM 优化章节统一使用同一批矩阵形状做横向比较，避免不同章节使用不同 shape 导致结果不可比。

统一 shape set 覆盖：

| 类别 | 形状 |
|------|------|
| square | `2048x2048x2048`, `4096x4096x4096` |
| tall-skinny | `8192x512x2048`, `16384x256x2048` |
| wide-short | `512x8192x2048` |
| large-K | `2048x2048x8192`, `1024x1024x16384` |

评测输出：

- Ch.12~Ch.17：每章 notebook 内提供本章方法的具体数值表格
- Ch.18：提供 Ch.12~Ch.18 + cuBLAS 的全方法总表和一张 latency 汇总图
- 主指标：`latency_ms`，越低越好
- 辅助指标：`TFLOPS`、`vs cuBLAS`、`vs previous chapter`

`torch.matmul` 在这些对比中作为 cuBLAS wrapper 使用。
```

- [ ] **Step 2: Review README rendering**

Run:

```bash
git diff -- README.md
```

Expected: README no longer claims the Part 3 benchmark is based only on `M=N=2048, K=1024`; it now describes the shared shape set.

- [ ] **Step 3: Commit README changes if commits are authorized**

```bash
git add README.md
git commit -m "Document shared Part 3 benchmark protocol"
```

Expected: one commit modifying only README.

---

### Task 11: Final Validation

**Files:**
- Verify: `03_matmul_optimization/benchmark_utils.py`
- Verify: `tests/test_benchmark_utils.py`
- Verify: `03_matmul_optimization/12_matmul_smem.ipynb`
- Verify: `03_matmul_optimization/13_matmul_pipeline.ipynb`
- Verify: `03_matmul_optimization/14_matmul_splitk.ipynb`
- Verify: `03_matmul_optimization/15_matmul_swizzle.ipynb`
- Verify: `03_matmul_optimization/16_matmul_fp16_acc.ipynb`
- Verify: `03_matmul_optimization/17_matmul_tensorcore.ipynb`
- Verify: `03_matmul_optimization/18_matmul_ultimate.ipynb`
- Verify: `README.md`

- [ ] **Step 1: Run helper tests**

```bash
python -m unittest tests/test_benchmark_utils.py -v
```

Expected: all helper tests pass.

- [ ] **Step 2: Run Ch.18 full execution check**

```bash
time conda run -n triton_tutorial jupyter nbconvert --to notebook --execute 03_matmul_optimization/18_matmul_ultimate.ipynb --output /tmp/ch18_final_check.ipynb
```

Expected:

- command exits with code 0
- runtime is clearly below the current about-10-minute experience
- `cumulative_optimization_latency.png` exists
- executed notebook contains `all_methods_df` and `summary_df`

- [ ] **Step 3: Run one representative earlier notebook**

Use Ch.13 because it checks previous-chapter comparison logic:

```bash
conda run -n triton_tutorial jupyter nbconvert --to notebook --execute 03_matmul_optimization/13_matmul_pipeline.ipynb --output /tmp/ch13_final_check.ipynb
```

Expected: command exits with code 0 and outputs `chapter13_df` with the shared seven shapes.

- [ ] **Step 4: Check changed files**

```bash
git status --short
git diff --stat
```

Expected changed files include only:

```text
03_matmul_optimization/benchmark_utils.py
tests/test_benchmark_utils.py
03_matmul_optimization/12_matmul_smem.ipynb
03_matmul_optimization/13_matmul_pipeline.ipynb
03_matmul_optimization/14_matmul_splitk.ipynb
03_matmul_optimization/15_matmul_swizzle.ipynb
03_matmul_optimization/16_matmul_fp16_acc.ipynb
03_matmul_optimization/17_matmul_tensorcore.ipynb
03_matmul_optimization/18_matmul_ultimate.ipynb
03_matmul_optimization/cumulative_optimization_latency.png
README.md
```

- [ ] **Step 5: Create final commit if commits are authorized and previous task commits were skipped**

```bash
git add 03_matmul_optimization/benchmark_utils.py \
        tests/test_benchmark_utils.py \
        03_matmul_optimization/12_matmul_smem.ipynb \
        03_matmul_optimization/13_matmul_pipeline.ipynb \
        03_matmul_optimization/14_matmul_splitk.ipynb \
        03_matmul_optimization/15_matmul_swizzle.ipynb \
        03_matmul_optimization/16_matmul_fp16_acc.ipynb \
        03_matmul_optimization/17_matmul_tensorcore.ipynb \
        03_matmul_optimization/18_matmul_ultimate.ipynb \
        03_matmul_optimization/cumulative_optimization_latency.png \
        README.md

git commit -m "Standardize Part 3 GEMM benchmarks"
```

Expected: one commit containing the complete benchmark redesign if incremental commits were not used.

---

## Self-Review

Spec coverage:

- Shared shape set across Ch.12-Ch.18: covered by Task 1 and Tasks 2-9.
- Primary metric `latency_ms`: covered by Task 1 helper schema and Ch.18 Task 9 figure.
- One Ch.18 global figure: covered by Task 9.
- Per-chapter numeric tables: covered by Tasks 2-7 and Ch.18 table in Task 9.
- Runtime reduction for Ch.18: covered by Task 8.
- README update: covered by Task 10.
- Validation: covered by Task 11.

Placeholder scan:

- This plan does not use open-ended implementation placeholders.
- The only conditional branch is the explicit Ch.18 integration note for missing wrappers; it names exactly which wrappers to copy from which chapter notebooks.

Type and name consistency:

- Shared result fields are consistent between `BenchmarkResult`, `format_results`, per-chapter table display, and Ch.18 summary code.
- Shape names match the approved spec.
- Metric names use `latency_ms`, `tflops`, `speedup_vs_cublas`, and `speedup_vs_previous` consistently.
