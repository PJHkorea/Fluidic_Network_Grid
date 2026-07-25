"""
================================================================================
🌊 FNG V3 PRODUCTION CORE - INTEGRATED HARDWARE-NEURAL ACCELERATION LIBRARY
================================================================================
Distributed Infrastructure Layer optimized via the JAX/XLA Compiler.
Copyright (c) 2026 PJHkorea. All Rights Reserved. Distributed under Apache 2.0.

This package serves as a high-performance infrastructure acceleration library combining 
continuum fluid mechanics (Burgers' Equation) with low-level compiler optimization parameters 
(Shard-Map Overlapping) to rectify gradient turbulence and prevent catastrophic numerical 
divergence (NaN/INF explosion) inside hyperscale distributed training pipelines.
"""

import sys

# --------------------------------------------------------------------------
# ⚡ LAYER 1: Explicit Package Architecture Definition (Static Namespace Freezing)
# --------------------------------------------------------------------------
# Explicitly export core acceleration kernels, neural-co design adapters, and the PyTorch mega-bridge 
# from the root namespace to eliminate runtime package-traversal and module-lookup overheads.
from production_core.core_smoother_xla import execute_gradient_viscous_smoother
from production_core.math_guardrails import enforce_algebraic_safety_gate
from production_core.async_scheduler import compile_asynchronous_overlapping_pipeline
from production_core.transformer_fused import FngInterleavedLlamaAttention
from production_core.pytorch_mega_adapter import FngPyTorchMegaAdapter

# Enforce an explicit __all__ export whitelist signature to prevent heap pollution from unintended 
# wildcard imports (e.g., `from production_core import *`), safeguarding interpreter-level memory 
# invariants and structural integrity.
__all__ = [
    "execute_gradient_viscous_smoother",
    "enforce_algebraic_safety_gate",
    "compile_asynchronous_overlapping_pipeline",
    "FngInterleavedLlamaAttention",
    "FngPyTorchMegaAdapter",
]

# --------------------------------------------------------------------------
# 🏛️ LAYER 2: Ecosystem Versioning & Defensive Spec Management
# --------------------------------------------------------------------------
__version__ = "3.0.0-production"
__author__ = "PJHkorea"
__license__ = "Apache-2.0"

def _audit_runtime_environment():
    """
    [SYSTEM AUDIT RUNTIME]
    Eagerly detects the availability and topological alignment of the distributed JAX/XLA 
    computing backend during the initial package initialization phase to intercept downstream 
    hardware allocation exceptions and eliminate infrastructure runtime downtime.
    """
    try:
        import jax
        # Intercept and alert upon hardware misalignments where JAX binds to host CPU slots instead of accelerator engines
        backend = jax.default_backend()
        if backend == "cpu":
            print("⚠️ [FNG WARNING]: JAX framework bound to host CPU slots. HPC Shard-Map Overlapping requires hardware accelerators (GPU/TPU).")
    except ImportError:
        raise ImportError(
            "❌ [FNG CRITICAL ERROR]: JAX infrastructure not detected in active workspace. "
            "Please install 'jax' and 'jaxlib' to compile and execute this high-performance distributed core."
        )

# Execute the bare-metal hardware and backend integrity audit immediately upon package load
_audit_runtime_environment()
