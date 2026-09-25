# 1. CI Verifies the CPU Path Only

Date: 2026-09-25

## Status

Accepted

## Context

Every milestone must run on Apple Silicon (PyTorch MPS) and NVIDIA GPUs (PyTorch CUDA), selected by a single `device` setting. M1–M19 use only PyTorch operators, so the same code also runs on CPU.

We want CI on GitHub Actions to check each change. GitHub-hosted runners limit what CI can check:

- `ubuntu-latest` runners have no GPU. Only the CPU device is available.
- macOS runners are virtual machines. MPS is not reliably available, so a CI job cannot depend on it.
- CUDA needs GPU runners, which are paid, or a self-hosted runner, which someone must maintain.

Paid GPU runners would catch CUDA problems, and a self-hosted Mac would catch MPS problems. Both cost money or upkeep that a learning repo does not justify.

## Decision

CI runs on `ubuntu-latest` and verifies the CPU path only.

- Tests run with `device="cpu"`.
- CI installs the CPU build of PyTorch, not the default CUDA build from PyPI.
- MPS and CUDA are verified by hand on local machines before a milestone is marked done.

## Consequences

- CI is free and fast, and it catches logic bugs that do not depend on the device.
- CI does not catch device-specific bugs. The README lists where the platforms differ:
  - Device synchronization before reading a timer (M10)
  - The memory budget used to size the KV pool (M15)
  - How much CPU/GPU overlap is possible (M19)
  - Custom Metal and Triton kernels (M20, M21)
  - How KV moves between processes (M22)
- Milestones that touch these areas need a manual check on MPS and CUDA.
- Tests must not assume a GPU. They must pass on CPU, for example with bf16 on CPU.
- `pyproject.toml` must let CI install CPU-only PyTorch without making CUDA machines install it too.
- If the repo later needs automatic GPU checks, for example for the M20 and M21 kernels, revisit this decision.
