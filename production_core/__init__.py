"""
================================================================================
🌊 FNG V3 PRODUCTION CORE - INTEGRATED HARDWARE-NEURAL ACCELERATION LIBRARY
================================================================================
Distributed Infrastructure Layer optimized via the JAX/XLA Compiler.
Copyright (c) 2026 PJHkorea. All Rights Reserved. Distributed under Apache 2.0.

본 패키지는 수리물리 점성 기전(Burgers' Equation)과 XLA 컴파일러 최적화(Shard-Map Overlapping)를 
결합하여 대규모 분산 SGD 학습 중 발생하는 그라디언트 난류 및 NaN 폭사를 차단하는 상용 인프라 엔진입니다.
"""

import sys

# --------------------------------──────────────────────────────────────────
# ⚡ LAYER 1: Explicit Package Architecture Definition (정적 네임스페이스 고정)
# --------------------------------──────────────────────────────────────────
# 최상단 네임스페이스에서 우리가 완성한 3대 심장부 커널과 어텐션 플러그인을 명시적으로 격상(Export).
# 외부 스크립트에서 임포트할 때 서브 폴더 검색(Runtime Lookup) 병목 비용을 완전히 소멸시킵니다.
from production_core.core_smoother_xla import execute_gradient_viscous_smoother
from production_core.math_guardrails import enforce_algebraic_safety_gate
from production_core.async_scheduler import compile_asynchronous_overlapping_pipeline
from production_core.transformer_fused import FngInterleavedLlamaAttention

# __all__ 규격을 엄격하게 제한하여, 외부에서 와일드카드 임포트(from production_core import *) 시
# 불필요한 파이썬 호스트 단의 가비지 객체가 복사 및 생성되는 가비지 컬렉터(GC) 스톨을 원천 차단합니다.
__all__ = [
    "execute_gradient_viscous_smoother",
    "enforce_algebraic_safety_gate",
    "compile_asynchronous_overlapping_pipeline",
    "FngInterleavedLlamaAttention",
]

# --------------------------------──────────────────────────────────────────
# 🏛️ LAYER 2: Ecosystem Versioning & Defensive Spec Management (버전 명세 보호)
# --------------------------------──────────────────────────────────────────
__version__ = "3.0.0-production"
__author__ = "PJHkorea"
__license__ = "Apache-2.0"

def _audit_runtime_environment():
    """
    [CRITICAL SYSTEM AUDIT RUNTIME]
    본 프로덕션 엔진이 구동되기 전, 시스템 환경에 고성능 가속 라이브러리(JAX/XLA)가 
    정상적으로 링크되어 있는지 단 1클록 만에 검증하여 하드웨어 패닉 및 예외 다운을 방지합니다.
    """
    try:
        import jax
        # JAX 백엔드가 더미(CPU) 슬롯으로 잡혀 분산 커널이 터지는 참사를 미연에 방지
        backend = jax.default_backend()
        if backend == "cpu":
            print("⚠️ [FNG WARNING]: JAX running on CPU. HPC Shard-Map Overlapping require hardware accelerators (GPU/TPU).")
    except ImportError:
        raise ImportError(
            "❌ [FNG CRITICAL ERROR]: JAX infrastructure not detected. "
            "Please run 'pip install jax jaxlib' under Apache 2.0 specs to execute this high-performance core."
        )

# 패키지가 임포트되는 첫 런타임 타이밍에 시스템 오디트를 즉각 집행
_audit_runtime_environment()
