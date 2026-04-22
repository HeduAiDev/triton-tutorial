import importlib.util
import math
import pathlib
import sys
import unittest
import warnings

import torch

ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "03_matmul_optimization" / "benchmark_utils.py"
spec = importlib.util.spec_from_file_location("benchmark_utils", MODULE_PATH)
benchmark_utils = importlib.util.module_from_spec(spec)
assert spec.loader is not None
sys.modules[spec.name] = benchmark_utils
spec.loader.exec_module(benchmark_utils)


class BenchmarkUtilsTest(unittest.TestCase):
    def test_shape_set_is_fixed_and_categorized(self):
        shapes = benchmark_utils.BENCHMARK_SHAPES
        self.assertEqual(
            [(s.name, s.category, s.M, s.N, s.K) for s in shapes],
            [
                ("square-2k", "square", 2048, 2048, 2048),
                ("square-4k", "square", 4096, 4096, 4096),
                ("tall-8k-x-512", "tall-skinny", 8192, 512, 2048),
                ("tall-16k-x-256", "tall-skinny", 16384, 256, 2048),
                ("wide-512-x-8k", "wide-short", 512, 8192, 2048),
                ("largeK-2k-x-8k", "large-K", 2048, 2048, 8192),
                ("largeK-1k-x-16k", "large-K", 1024, 1024, 16384),
            ],
        )
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

    def test_benchmark_method_marks_failed_correctness_as_missing(self):
        shape = benchmark_utils.BenchmarkShape("shape", "square", 2, 2, 2)
        a = torch.zeros((2, 2), dtype=torch.float16)
        b = torch.zeros((2, 2), dtype=torch.float16)
        c_ref = torch.zeros((2, 2), dtype=torch.float16)

        result = benchmark_utils.benchmark_method(
            "bad",
            lambda x, y: torch.ones((2, 2), dtype=torch.float16),
            shape,
            a,
            b,
            c_ref=c_ref,
            atol=0.0,
        )

        self.assertFalse(result.passed)
        self.assertTrue(math.isnan(result.latency_ms))
        self.assertTrue(math.isnan(result.tflops))
        self.assertEqual(result.max_err, 1.0)

    def test_max_abs_error_rejects_broadcastable_shape_mismatch(self):
        candidate = torch.zeros((1, 2), dtype=torch.float16)
        reference = torch.zeros((2, 2), dtype=torch.float16)
        self.assertEqual(benchmark_utils.max_abs_error(candidate, reference), math.inf)

    def test_time_cuda_validates_timing_counts_before_cuda_work(self):
        a = torch.zeros((1, 1), dtype=torch.float16)
        b = torch.zeros((1, 1), dtype=torch.float16)
        fn = lambda x, y: x

        with self.assertRaisesRegex(ValueError, "warmup must be >= 0"):
            benchmark_utils.time_cuda(fn, a, b, warmup=-1, rep=1)
        with self.assertRaisesRegex(ValueError, "rep must be > 0"):
            benchmark_utils.time_cuda(fn, a, b, warmup=0, rep=0)

    def test_benchmark_method_warns_on_runtime_failure(self):
        shape = benchmark_utils.BenchmarkShape("shape", "square", 2, 2, 2)
        a = torch.zeros((2, 2), dtype=torch.float16)
        b = torch.zeros((2, 2), dtype=torch.float16)

        def fail(_a, _b):
            raise RuntimeError("bad launch")

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = benchmark_utils.benchmark_method("bad", fail, shape, a, b)

        self.assertFalse(result.passed)
        self.assertTrue(math.isnan(result.latency_ms))
        self.assertTrue(math.isnan(result.tflops))
        self.assertIn("bad failed on shape: bad launch", str(caught[0].message))

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
