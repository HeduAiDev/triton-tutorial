# Skill Improvement Notes — Triton Tutorial Ch27 (PyPTO Comparison)

> Auto-maintained during tutorial generation. Each entry is a concrete,
> actionable suggestion for improving the `reimpl-tutorial` skill.

## Process

### Continuation Mode with mixed notebook creation patterns
- **Phase encountered:** 1
- **Current behavior:** Skill mandates Node.js builder scripts for ALL notebooks
- **Problem:** Existing project (26 chapters) uses direct `.ipynb` editing via NotebookEdit. Switching to builder scripts for just Ch27 creates inconsistency in the project.
- **Suggested fix:** In Continuation Mode, add a step: "Detect existing notebook creation pattern (builder scripts vs direct edit). If the existing project does NOT use builder scripts, consider adopting the existing pattern for consistency, OR create the builder script but document the inconsistency."
- **Evidence:** Ch27 uses builder script while Ch01-Ch26 were created with direct NotebookEdit

### Comparative tutorial with asymmetric runnability
- **Phase encountered:** 3
- **Current behavior:** Skill assumes all code is either runnable or explanatory-only (binary choice)
- **Problem:** In cross-platform comparison (GPU vs NPU), one framework's code is runnable and the other isn't. We needed a hybrid mode: Triton code cells run normally, PyPTO code is shown in markdown code blocks (explanatory).
- **Suggested fix:** Add a "hybrid runnability" option for comparative tutorials where one project runs locally and the other doesn't. The verification pattern should only cover the runnable side.
- **Evidence:** Ch27 has runnable Triton cells but PyPTO code in markdown blocks

## Changelog

_To be updated when skill improvements are applied._
