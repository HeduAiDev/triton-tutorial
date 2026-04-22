from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Callable, Iterable
import warnings

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


def time_cuda(
    fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    a: torch.Tensor,
    b: torch.Tensor,
    warmup: int = DEFAULT_WARMUP,
    rep: int = DEFAULT_REP,
) -> float:
    if warmup < 0:
        raise ValueError("warmup must be >= 0")
    if rep <= 0:
        raise ValueError("rep must be > 0")

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
    if candidate.shape != reference.shape:
        return math.inf
    return (candidate - reference).abs().max().item()


def benchmark_method(
    method_name: str,
    fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    shape: BenchmarkShape,
    a: torch.Tensor,
    b: torch.Tensor,
    c_ref: torch.Tensor | None = None,
    warmup: int = DEFAULT_WARMUP,
    rep: int = DEFAULT_REP,
    atol: float = 2.0,
) -> BenchmarkResult:
    try:
        candidate = fn(a, b)
        max_err = max_abs_error(candidate, c_ref) if c_ref is not None else math.nan
        passed = c_ref is None or max_err <= atol
        latency_ms = math.nan
        tflops = math.nan
        if passed:
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
    except Exception as exc:
        warnings.warn(
            f"{method_name} failed on {shape.name}: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
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


def add_relative_columns(
    df: pd.DataFrame,
    cublas_method: str = "cuBLAS",
    previous_method_by_method: dict[str, str] | None = None,
) -> pd.DataFrame:
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
        rows.append(
            {
                "shape_name": shape_name,
                "category": best["category"],
                "best_method": best["method"],
                "best_triton_ms": float(best["latency_ms"]),
                "cublas_ms": cublas_ms,
                "best_triton_vs_cublas": speedup(cublas_ms, float(best["latency_ms"])),
            }
        )
    return pd.DataFrame(rows)
