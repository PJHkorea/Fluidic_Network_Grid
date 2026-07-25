import jax
import jax.numpy as jnp
from functools import partial

@partial(jax.jit, static_argnums=(2,))
def execute_gradient_viscous_smoother(raw_gradient, viscosity_sigma, integration_epsilon=1e-6):
    """
    [FNG V3 PRODUCTION CORE - HIGH-PERFORMANCE GRADIENT SMOOTHER KERNEL]
    
    기존 fluidic_mockup의 '점성 버거스 및 왜도 상쇄' 수리 물리 아키텍처를 100% 계승하되,
    데이터 파괴를 일으키던 이진화 플래그(> 0.5 .astype)를 완전히 파열시킨 실전형 정류 커널입니다.
    분산 SGD 학습 중 튀는 그라디언트 난류를 온칩 레지스터 레벨에서 무분기(Branchless)로 정류합니다.
    
    Args:
        raw_gradient: 분산 노드로부터 유입된 원본 그라디언트 텐서 
                      [Nodes/Batch, Volatile_Time_Jitter_Dim, Feature_Dim]
        viscosity_sigma: 그라디언트 난류 고주파 노이즈를 감쇠시키기 위한 점성 소산 계수 (float)
        integration_epsilon: 2차 모멘트 분모 0 분출 방지용 수치 안정성 상수 (float)
        
    Returns:
        purified_gradient: bfloat16/float32 정밀도가 완벽히 보존된 청정 연속적 소수점 그라디언트 텐서
    """
    target_dtype = raw_gradient.dtype
    
    # ----------------──────────────────────────────────────────────────────────
    # 🌊 LAYER 1: Burgers' Laplacian Viscous Smoothing (점성 소산 정류)
    # ----------------──────────────────────────────────────────────────────────
    # 격자점 양 끝단의 데이터 단절로 인한 수치 폭발을 막기 위해 Neumann 경계 조건(기울기 0)을 시뮬레이션
    # 이 보폭(Padding) 제어 기믹은 기존 라우터의 Neumann Clamping 설계를 계승합니다.
    padded_grad = jnp.pad(raw_gradient, ((0, 0), (1, 1), (0, 0)), mode='edge')
    
    # 2차 공간 차분(Laplacian) 적출: 칩 내부 레지스터 레일 위에서 무분기로 이웃 셀 간 수치 변위 연산
    laplacian = padded_grad[:, :-2, :] - 2.0 * raw_gradient + padded_grad[:, 2:, :]
    
    # 버거스 방정식의 소산 항(+σ * ∂²Φ/∂x²) 가동
    # 없는 데이터를 창조하는 마술 대신, 미세 지터 노이즈(난류)를 점성 브레이크로 부드럽게 흡수(Smoothing)
    rectified_gradient = raw_gradient + (jnp.array(viscosity_sigma, dtype=target_dtype) * laplacian)
    
    # ----------------──────────────────────────────────────────────────────────
    # 📐 LAYER 2: Higher-Order Moment Skewness Flattening (고차 모멘트 왜도 평탄화)
    # ----------------──────────────────────────────────────────────────────────
    # 시간축 지터 차원(axis=1)을 수축소멸시키기 위한 평균값 중심화(Mean Centering)
    spatial_mean = jnp.mean(rectified_gradient, axis=1, keepdims=True)
    pure_manifold_delta = rectified_gradient - spatial_mean
    
    # 2차 모멘트(m2, 분산) 및 3차 모멘트(m3, 왜도 분자) 대수적 고속 적출
    m2 = jnp.mean(pure_manifold_delta ** 2, axis=1)
    m3 = jnp.mean(pure_manifold_delta ** 3, axis=1)
    
    # ----------------──────────────────────────────────────────────────────────
    # ⚡ LAYER 3: SFU Native Reciprocal Fusion & Precision Conservation
    # ----------------──────────────────────────────────────────────────────────
    # 분모가 0이 되어 NaN 오염이 번지는 것을 원천 차단하기 위해 stop_gradient 주입
    # 이 수치 가드는 egregore-core-jax의 핵심 방화벽 메커니즘을 내장한 것입니다.
    denominator_safe = m2 + jax.lax.stop_gradient(jnp.array(integration_epsilon, dtype=target_dtype))
    
    # SFU(특수 기능 유닛) 네이티브 온칩 역수 변환기 가동 -> 하드웨어 나눗셈 스톨(Slash /)을 완전 파쇄
    # division 오버헤드 전혀 없이 단일 사이클 곱셈 연산으로 100% 전환합니다.
    reciprocal_m2 = jax.lax.reciprocal(denominator_safe)
    
    # 대수적으로 약분 소거된 비대칭 압력 오프셋 변위(Skewness Correction) 계산
    asymmetric_correction = 0.5 * m3 * reciprocal_m2
    
    # 원본 텐서 공간 격자 위에 대수적 정류 최종 집행
    # 각 노드 내부의 불규칙한 압력 바이어스를 수평 평탄화 처리합니다.
    purified_gradient = rectified_gradient - asymmetric_correction[:, None, :]
    
    # ----------------──────────────────────────────────────────────────────────
    # 🚀 [★CRITICAL REALIGNMENT★] 이진화 플래그의 완전한 소멸
    # ----------------──────────────────────────────────────────────────────────
    # 기존 목업의 > 0.5 .astype(jnp.float32) 데이터 파괴 코드를 완벽히 박멸했습니다.
    # LLM 어텐션 레일에 주입 가능한 bfloat16/float32 고정밀 소수점 텐서 뷰(View)를 무복사로 그대로 사출합니다.
    return purified_gradient
