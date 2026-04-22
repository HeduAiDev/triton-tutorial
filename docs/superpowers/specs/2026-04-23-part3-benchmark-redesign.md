# Part 3 Benchmark Redesign

## Goal

Redesign the Part 3 GEMM benchmarking so that all major optimization chapters are evaluated on the same fixed shape set, making cross-chapter comparisons fair and reproducible.

The redesign has four explicit goals:

1. Every benchmarked method uses the same shape set.
2. The primary comparison metric is latency in milliseconds (`latency_ms`).
3. `03_matmul_optimization/18_matmul_ultimate.ipynb` provides one global summary figure across all methods.
4. Every chapter notebook from Ch.12 to Ch.18 contains at least one numeric table with concrete benchmark values.

A second hard requirement is runtime control: the default execution path of `03_matmul_optimization/18_matmul_ultimate.ipynb` must be reduced from the current about-10-minute scale on RTX PRO 6000 hardware to a notebook-friendly runtime budget.

## Scope

### In scope

- Standardize benchmark shapes across Ch.12 to Ch.18.
- Add per-chapter numeric benchmark tables.
- Replace the current Ch.18 cumulative visualization with a single global latency summary figure.
- Reduce the runtime cost of the default Ch.18 execution path by shrinking benchmark and config space.
- Update README wording so it reflects the new benchmark protocol.

### Out of scope

- Rewriting the conceptual teaching flow of each notebook.
- Turning every notebook into a full all-method benchmark harness.
- Running large autotune searches as part of the default notebook execution path.
- Hiding or smoothing over cases where an optimization is slower on some shapes.

## Benchmark Protocol

### Methods included in the global comparison

The global comparison in `03_matmul_optimization/18_matmul_ultimate.ipynb` will include the following representative methods:

1. `Ch.12 smem`
2. `Ch.13 pipeline`
3. `Ch.14 split-k`
4. `Ch.15 swizzle`
5. `Ch.16 mixed-precision / accumulator variant`
6. `Ch.17 tensorcore-focused variant`
7. `Ch.18 ultimate`
8. `cuBLAS`

These entries represent the final teaching-stage implementation from each chapter, not every experimental variant inside the notebook.

`Ch.14 split-k` remains in the comparison even if it loses on some shapes. That is part of the teaching value: Split-K is shape-sensitive and should be presented honestly rather than filtered to only favorable cases.

### Fixed benchmark shape set

The shared benchmark shape set will contain seven representative shapes:

```python
benchmark_shapes = [
    ("square-2k",      "square",       2048,  2048,  2048),
    ("square-4k",      "square",       4096,  4096,  4096),
    ("tall-8k-x-512",  "tall-skinny",  8192,   512,  2048),
    ("tall-16k-x-256", "tall-skinny", 16384,   256,  2048),
    ("wide-512-x-8k",  "wide-short",    512,  8192,  2048),
    ("largeK-2k-x-8k", "large-K",      2048,  2048,  8192),
    ("largeK-1k-x-16k", "large-K",     1024,  1024, 16384),
]
```

This set is intentionally small enough to keep notebook execution practical while still covering:

- standard square GEMM behavior
- tall-skinny output shapes
- wide-short output shapes
- large-K pressure where data movement and pipelining matter more

All benchmark tables and the Ch.18 summary figure must use this same shape set.

### Primary and secondary metrics

Primary metric:

- `latency_ms`

Secondary metrics:

- `tflops`
- `speedup_vs_cublas`
- when appropriate, `speedup_vs_previous_chapter`

The main figure will use `latency_ms` only. Tables may include the secondary metrics for interpretability.

## Presentation Design

### Ch.18 global summary

`03_matmul_optimization/18_matmul_ultimate.ipynb` will contain:

1. One all-method result table covering every method and every shared shape.
2. One grouped bar figure with:
   - X axis = shape name
   - Y axis = `latency_ms`
   - color = method
3. One compact textual or tabular summary indicating:
   - fastest method per shape
   - `cuBLAS` latency
   - best Triton latency
   - ratio of best Triton to `cuBLAS`

There should be exactly one global figure in Ch.18 for this benchmark redesign. The figure is responsible for trend overview; the tables are responsible for concrete values.

### Per-chapter numeric tables

Every notebook from Ch.12 through Ch.18 will include at least one numeric benchmark table based on the shared shape set.

Expected comparison scope by chapter:

1. `12_matmul_smem.ipynb`
   - `smem` vs `cuBLAS`
2. `13_matmul_pipeline.ipynb`
   - `pipeline` vs `smem` vs `cuBLAS`
3. `14_matmul_splitk.ipynb`
   - `split-k` vs `pipeline` vs `cuBLAS`
4. `15_matmul_swizzle.ipynb`
   - `swizzle` vs `pipeline` vs `cuBLAS`
5. `16_matmul_fp16_acc.ipynb`
   - chapter variant vs chapter baseline vs `cuBLAS`
6. `17_matmul_tensorcore.ipynb`
   - chapter variant vs prior variant vs `cuBLAS`
7. `18_matmul_ultimate.ipynb`
   - all-method table + global figure

Each per-chapter table should include at least:

- `shape`
- `latency_ms`
- `tflops`
- `vs cuBLAS`

Where the chapter has a clear predecessor baseline, it should also include:

- `vs previous chapter`

The chapter tables are the authoritative location for concrete numbers in that chapter. Additional figures are not required in every notebook.

## Runtime Budget And Configuration Reduction

### Problem statement

The current `03_matmul_optimization/18_matmul_ultimate.ipynb` default path is too expensive: a single run can take about 10 minutes even on RTX PRO 6000 hardware. That is too slow for a tutorial notebook that readers are expected to execute.

### Required changes

The redesign must reduce runtime by changing the default benchmark path:

1. Shrink the `ultimate` config space.
   - Keep only representative autotune configs.
   - Remove redundant tile / warp / stage combinations that mostly increase search cost.
2. Do not perform large autotune sweeps during the global benchmark phase.
   - The benchmark should evaluate one selected representative implementation per chapter.
3. Keep the shared shape set limited.
   - Total shape count stays at seven entries.
4. Reduce timing repetition to a tutorial-friendly level.
   - Use enough warmup and repetition for stability, but not enough to dominate notebook runtime.
5. Separate teaching defaults from exhaustive tuning.
   - The notebook default path should be educational and practical.
   - Any more exhaustive tuning path, if retained at all, should be clearly non-default.

### Runtime target

The final implementation should reduce the default execution time of Ch.18 substantially below the current 10-minute scale. The exact number can depend on hardware and notebook environment, but the result should clearly no longer feel like an exhaustive search notebook.

## Data Collection Rules

### Correctness

Every compared method must pass a correctness check against `torch.matmul` before or during benchmarking.

If a method is not applicable to a shape or fails to run correctly:

- record the result as missing (`NaN` or equivalent)
- skip plotting that bar
- mark it clearly in the chapter or summary table

Do not silently replace a failed method with another implementation.

### Result schema

Collected results should map cleanly into a structured table or DataFrame with fields like:

```python
{
    "shape_name": "square-2k",
    "category": "square",
    "M": 2048,
    "N": 2048,
    "K": 2048,
    "method": "Ch.13 pipeline",
    "latency_ms": 0.123,
    "tflops": 139.6,
    "speedup_vs_cublas": 0.94,
}
```

This result schema should be shared conceptually across all chapter tables and the Ch.18 summary.

## File-Level Change Plan

### Primary notebook changes

- `03_matmul_optimization/12_matmul_smem.ipynb`
  - add numeric benchmark table on shared shape set
- `03_matmul_optimization/13_matmul_pipeline.ipynb`
  - add numeric benchmark table on shared shape set
- `03_matmul_optimization/14_matmul_splitk.ipynb`
  - add numeric benchmark table on shared shape set
- `03_matmul_optimization/15_matmul_swizzle.ipynb`
  - add numeric benchmark table on shared shape set
- `03_matmul_optimization/16_matmul_fp16_acc.ipynb`
  - add numeric benchmark table on shared shape set
- `03_matmul_optimization/17_matmul_tensorcore.ipynb`
  - add numeric benchmark table on shared shape set
- `03_matmul_optimization/18_matmul_ultimate.ipynb`
  - add shared benchmark harness
  - add all-method result table
  - replace existing cumulative visualization with one global latency figure
  - reduce default config / benchmark cost

### Documentation changes

- `README.md`
  - update Part 3 benchmark description
  - note that Ch.18 contains the global summary figure
  - note that each chapter contains numeric tables on the same shared shape set

## Validation And Acceptance Criteria

The redesign is complete only when all of the following are true:

1. Ch.12 through Ch.18 all expose numeric benchmark tables using the same shared shape set.
2. Ch.18 contains exactly one global summary figure for the cross-method comparison.
3. The main visual metric in Ch.18 is `latency_ms`.
4. Ch.18 also provides a concrete all-method numeric summary.
5. Results are honest about shape sensitivity and do not hide regressions.
6. The default execution path of `03_matmul_optimization/18_matmul_ultimate.ipynb` is substantially faster than the current about-10-minute experience on RTX PRO 6000.
7. README language matches the new protocol.

## Design Notes

This redesign intentionally separates the responsibilities of figures and tables:

- the single Ch.18 figure shows global trends
- per-chapter tables provide exact numbers for chapter-specific conclusions

That split avoids repeated chart noise across notebooks while still making every chapter quantitatively useful and cross-checkable.
