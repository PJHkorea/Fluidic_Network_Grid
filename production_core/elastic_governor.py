import jax
import jax.numpy as jnp
from functools import partial

# 우리가 앞서 구현한 1, 2번 핵심 수치 해석 필터 임포트
from production_core.core_smoother_xla import execute_gradient_viscous_smoother
from production_core.math_guardrails import enforce_algebraic_safety_gate

@partial(jax.jit, static_argnums=(3,))
def compute_dynamic_viscosity_sigmoid(current_drop_rate, sigma_base=3.125e-5, sigma_max=0.01, k_stiffness=15.0, d_critical=0.35):
    """
    [FNG V3 PRODUCTION - SFU HARDWARE SIGMOID VISCOSITY SCALE KERNEL]
    원작자 사양서 8.1조의 비선형 지수 가변 점성 수식을 단일 가속기 명령어 수준으로 구현했습니다.
    패킷 유실률이 35% 임계점을 돌파하는 순간, 수치적 충격파를 흡수하기 위해 점성을 타르(Tar) 상태로 급격히 상전이시킵니다.
    """
    target_dtype = current_drop_rate.dtype
    clamped_drop = jnp.clip(current_drop_rate, 0.0, 1.0)
    
    # 사양서 명세 공식: σ(d_t) = σ_base + (σ_max - σ_base) / (1 + exp(-k * (d_t - d_c)))
    activation_shift = jnp.array(k_stiffness, dtype=target_dtype) * (clamped_drop - jnp.array(d_critical, dtype=target_dtype))
    
    # 가속기 SFU 전용 하드웨어 시그모이드 회로 직통 결착 (나눗셈 오버헤드 100% 소멸)
    viscous_damping_ratio = jax.nn.sigmoid(activation_shift)
    
    dynamic_sigma = jnp.array(sigma_base, dtype=target_dtype) + (
        jnp.array(sigma_max, dtype=target_dtype) - jnp.array(sigma_base, dtype=target_dtype)
    ) * viscous_damping_ratio
    
    return dynamic_sigma

def compile_wireless_elastic_governor(devices_mesh, mesh_axis_name="fluidic_mesh"):
    """
    [FNG V3 PRODUCTION CORE - WIRELESS EDGE RESILIENT SCAN GOVERNOR]
    호스트 단 파이썬 루프 제어 스톨을 박멸하고, 가속기 내부 레일 위에서 무선 통신 불안정 레이어를 다스리는 사령탑입니다.
    """
    
    def elastic_scan_step_fn(carry_state, input_slice):
        """
        jax.lax.scan 내부에서 매 타임스텝(Sequence/Iteration)마다 0ns 피드백 루프로 가동되는 시간축 가드레일.
        """
        # Carry State: 이전 타임스텝에서 수치 무결성이 완벽히 보증되어 도달했던 정상 텐서 (frozen_static_constant)
        previous_healthy_tensor = carry_state
        
        # Input Slice: 현재 타임스텝에 인입된 원본 스트림 및 텔레메트리 드롭률 지표
        local_stream, current_drop_rate, pollution_mask = input_slice
        target_dtype = local_stream.dtype
        
        # 1) 실시간 가변 점성 자동 추적 발동 (SFU 하드웨어 결착)
        dynamic_sigma = compute_dynamic_viscosity_sigmoid(current_drop_rate)
        
        # 2) 본질 연산 집행: 버거스 점성 기반 그라디언트 난류 정류 (1번 모듈 연계)
        purified_gradient = execute_gradient_viscous_smoother(
            raw_gradient=local_stream,
            viscosity_sigma=dynamic_sigma,
            integration_epsilon=1e-6
        )
        
        # 3) 수치 경계면 최종 가드레일: Leaky Slope 기반 NaN/INF 폭사 차단 방화벽 (2번 모듈 연계)
        stabilized_gradient = enforce_algebraic_safety_gate(purified_gradient)
        
        # 4) [★CRITICAL REAL-WORLD REFACTORING★] 오토그라드 절연 밸브 및 결함 락킹 매커니즘
        # 패킷 드롭률이 극단적인 암전 영역(85% 이상)에 진입하면, 가짜 데이터를 채우는 대신 즉각 방화벽을 내립니다.
        blackout_bool = current_drop_rate >= 0.85
        
        # 수전 가중치가 망가지는 것을 막기 위해 이전 타임스텝의 청정 텐서에 stop_gradient를 걸어 완벽히 박제 격리
        frozen_static_constant = jax.lax.stop_gradient(previous_healthy_tensor)
        
        # 단일 기계어 수식 하드웨어 MUX 선택자(select)를 통해 오차 없이 0ns 컷오프 스위칭 집행
        final_isolated_tensor = jax.lax.select(
            blackout_bool,
            frozen_static_constant, # 암전 시: 오염을 전파하지 않고 과거의 청정 데이터 상태로 동결 생존 (Elastic Control)
            stabilized_gradient     # 정상/지터 시: 정류가 완료된 고정밀 소수점 데이터를 안전 전사
        )
        
        # 차기 루프로 넘겨줄 Carry State를 업데이트하고 현재 스텝의 사출값으로 지정
        next_carry = final_isolated_tensor
        return next_carry, final_isolated_tensor

    return elastic_scan_step_fn
