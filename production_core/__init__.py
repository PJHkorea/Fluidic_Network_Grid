"""
================================================================================
🌊 FNG V3 PRODUCTION CORE - INTEGRATED HARDWARE-NEURAL ACCELERATION LIBRARY
================================================================================
Distributed Infrastructure Layer optimized via the JAX/XLA Compiler.
Copyright (c) 2026 PJHkorea. All Rights Reserved. Distributed under Apache 2.0.

본 패키지는 수리물리 점성 기전(Burgers' Equation)과 XLA 컴파일러 최적화(Shard-Map Overlapping)를 
결합하여 분산 학습 중 발생하는 그라디언트 난류와 NaN 발산을 안정화하는 인프라 가속 라이브러리입니다.
"""

import sys

# --------------------------------------------------------------------------
# ⚡ LAYER 1: Explicit Package Architecture Definition (정적 네임스페이스 고정)
# --------------------------------------------------------------------------
# 최상단 네임스페이스에서 코어 커널, 어텐션 어댑터 및 PyTorch 프레임워크 어댑터를 명시적으로 노출(Export).
# 외부 스크립트 임포트 시 서브 모듈 탐색(Runtime Lookup) 오버헤드를 최소화합니다.
from production_core.core_smoother_xla import execute_gradient_viscous_smoother
from production_core.math_guardrails import enforce_algebraic_safety_gate
from production_core.async_scheduler import compile_asynchronous_overlapping_pipeline
from production_core.transformer_fused import FngInterleavedLlamaAttention
from production_core.pytorch_mega_adapter import FngPyTorchMegaAdapter

# __all__ 명세를 엄격히 제한하여, 와일드카드 임포트(from production_core import *) 발생 시
# 불필요한 가비지 객체 생성을 차단하고 인터프리터 런타임의 메모리 정합성을 유지합니다.
__all__ = [
    "execute_gradient_viscous_smoother",
    "enforce_algebraic_safety_gate",
    "compile_asynchronous_overlapping_pipeline",
    "FngInterleavedLlamaAttention",
    "FngPyTorchMegaAdapter",
]

# --------------------------------------------------------------------------
# 🏛️ LAYER 2: Ecosystem Versioning & Defensive Spec Management (버전 명세 보호)
# --------------------------------------------------------------------------
__version__ = "3.0.0-production"
__author__ = "PJHkorea"
__license__ = "Apache-2.0"

def _audit_runtime_environment():
    """
    [SYSTEM AUDIT RUNTIME]
    패키지 로드 시점에 분산 컴퓨팅 백엔드(JAX/XLA)가 정상적으로 연결 및 링크되어 있는지 
    사전 감지하여 하드웨어 가동 예외 및 다운타임 발생을 방지합니다.
    """
    try:
        import jax
        # JAX 백엔드가 가속기 대신 CPU 슬롯으로 오매핑되는 현상을 사전 감지
        backend = jax.default_backend()
        if backend == "cpu":
            print("⚠️ [FNG WARNING]: JAX running on CPU. HPC Shard-Map Overlapping requires hardware accelerators (GPU/TPU).")
    except ImportError:
        raise ImportError(
            "❌ [FNG CRITICAL ERROR]: JAX infrastructure not detected. "
            "Please install 'jax' and 'jaxlib' to execute this high-performance core."
        )

# 패키지가 임포트되는 첫 런타임 타이밍에 시스템 오디트를 즉각 집행
_audit_runtime_environment()

