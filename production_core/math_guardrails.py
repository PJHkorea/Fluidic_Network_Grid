import jax
import jax.numpy as jnp
from functools import partial

@partial(jax.jit, static_argnums=(2, 3))
def enforce_algebraic_safety_gate(purified_gradient, global_threshold=1e6, leaky_slope=1e-3, clean_baseline_val=0.0):
    """
    [FNG V3 PRODUCTION CORE - ALGEBRAIC SAFETY GATE & NAN FIREWALL]
    
    egregore-core-jax의 핵심 방화벽 메커니즘을 JAX/XLA 네이티브 기계어 레벨로 구현한 실전형 커널입니다.
    분산 학습 중 발생하는 수치 발산(NaN/INF) 및 임계치 초과 스파이크를 단 하나의 조건부 분기(if-else/JMP) 없이 
    원자적으로 하드 플러시하되, 미분 사슬의 절연을 막기 위해 경계면 바깥에 미세 기울기(Leaky Slope)를 주입합니다.
    
    Args:
        purified_gradient: core_smoother_xla를 통과하여 정류된 bfloat16/float32 소수점 그라디언트 텐서
        global_threshold: 수치적 파멸(Explosion)로 간주할 절대 임계값 (기본값: 1.0e6)
        leaky_slope: 임계치 초과 영역에서 역전파 미분 사슬을 살려두기 위한 미세 감쇠 기울기 (기본값: 0.001)
        clean_baseline_val: 완전 결함(NaN) 감지 시 리셋할 저전압 논리 베이스라인 부호 (기본값: 0.0)
        
    Returns:
        stabilized_gradient: 수치 안정성이 완벽히 확보되고 미분 선로가 보존된 청정 프로덕션 그라디언트 텐서
    """
    target_dtype = purified_gradient.dtype
    threshold_tensor = jnp.array(global_threshold, dtype=target_dtype)
    slope_tensor = jnp.array(leaky_slope, dtype=target_dtype)
    baseline_tensor = jnp.array(clean_baseline_val, dtype=target_dtype)

    # ----------------──────────────────────────────────────────────────────────
    # ⚡ STEP 1: Hardware-Native NaN / INF Artifact Capture (논리합 비트 예외 포획)
    # ----------------──────────────────────────────────────────────────────────
    # jnp.isnan 및 jnp.isinf를 조합하여 하부 실리콘 레벨의 예외 부호를 감지
    # 조건문 분기를 쓰지 않고, 오염된 비트 좌표를 1.0(True) 플래그 행렬로 적출합니다.
    invalid_mask = jnp.isnan(purified_gradient) | jnp.isinf(purified_gradient)
    invalid_mask_float = invalid_mask.astype(target_dtype)

    # 1차 원자적 정화: 결함 부호(NaN/INF)가 터진 자리는 즉각 청정 베이스라인 논리 레일(0.0f)로 플러시
    # 하드웨어 MUX 선택자 기전(jax.lax.select)을 활용해 0ns 연산 평탄화를 달성합니다.
    purged_step_1 = jax.lax.select(invalid_mask, jnp.full_like(purified_gradient, baseline_tensor), purified_gradient)

    # ----------------──────────────────────────────────────────────────────────
    # 🎛️ STEP 2: Leaky Slope Mapping for Extreme Spikes (미세 기울기 가드레일)
    # ----------------──────────────────────────────────────────────────────────
    # 절대값이 글로벌 임계값(1e6)을 초과하여 터지기 일보 직전인 익스트림 스파이크 좌표 추적
    abs_gradient = jnp.abs(purged_step_1)
    overflow_mask = abs_gradient > threshold_tensor

    # [★CRITICAL EGREGORE PIVOT★] 
    # 값을 단순히 clipping(jnp.clip)하여 상수로 굳혀버리면, 미분값이 0이 되어 역전파 학습이 영구 정지됩니다.
    # 이를 우회하기 위해 임계값을 넘은 초과 변위량(delta)에 미세 기울기(leaky_slope)를 곱해 그라디언트를 살려둡니다.
    sign_mask = jnp.sign(purged_step_1)
    excess_delta = abs_gradient - threshold_tensor
    
    # 임계치 영역 내부 = 원본 값 유지 / 임계치 영역 바깥 = Threshold + (Excess * leaky_slope) 부호 보존 합성
    leaky_clamped_value = sign_mask * (threshold_tensor + (excess_delta * slope_tensor))
    
    # 조건부 분기문(if-else)을 완벽히 소멸시킨 단일 기계어 수식 MUX 최종 결착
    stabilized_gradient = jax.lax.select(overflow_mask, leaky_clamped_value, purged_step_1)

    return stabilized_gradient
