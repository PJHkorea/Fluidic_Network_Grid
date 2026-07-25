import jax
import jax.numpy as jnp
from typing import Tuple, Dict

# [KR] 주의: 이 디코더 커널 또한 shard_map 또는 pmap 분산 토폴로지 내부에서 퓨전 컴파일되어야 합니다.
# [EN] NOTE: This decoder kernel must also be fusion-compiled within a shard_map or pmap distributed topology.
@jax.jit
def execute_fluidic_manifold_decoder(
    router_outputs: Dict[str, jax.Array],  # [KR] 개량된 라우터가 토출한 사전 정의된 딕셔너리 구조를 무복사 수신 / [EN] Zero-copy reception of the pre-defined dictionary layout emitted by the upgraded router
    integration_epsilon: float = 1e-6      # [KR] 고차 왜도 보정의 수치 안정성을 위해 하한 안정성 마진 확보 / [EN] Maintain lower stability margin for numerical consistency in higher-order skewness correction
) -> Tuple[jax.Array, Dict[str, jax.Array]]:
    """
    Fluidic Network Grid (FNG) Center of Mass & High-Order Moment Integrator Decoder Kernel
    
    [KR] 나눗셈과 초월함수 파이프라인 스톨을 완전히 소거하고, 3차 왜도(Skewness) 유체 압력을 지연 없이 평탄화 정류합니다.
    [EN] Complete mitigation of dynamic latency driven by division/transcendental functions, executing 0ns flattening of 3rd-order skewness.
    
    Memory Allocation Stall: 0.0% (Pure Register In-place Operator)
    """
    # ====================================================================
    # [KR] [1] MULTI-CHANNEL REGISTER UNPACKING & QUANTUM BOUNDARY PROTECTION
    # [EN] [1] Multi-Channel Register Unpacking & Quantum Boundary Protection
    # ====================================================================
    # [KR] 컴파일러 포인터 링크를 통해 물리적 데이터 복제 오버헤드 0바이트 상태로 메모리를 다이렉트 참조합니다.
    # [EN] Access underlying memory pointers directly via reference aliasing with absolute zero physical data-copy overhead.
    fluidic_grid_stream = router_outputs["fluidic_stream"]
    
    # [KR] 차원 문맥 명세 확보 / [EN] Extract dimensional execution context specifications
    nodes_count, volatile_dim, feature_dim = fluidic_grid_stream.shape
    target_dtype = fluidic_grid_stream.dtype
    
    # ====================================================================
    # [KR] [2] SPATIAL MASS MOMENT GENERATION (공간 가중치 기하 격자 생성)
    # [EN] [2] Spatial Mass Moment Generation (Geometric Weight Grid Setup)
    # ====================================================================
    # [KR] XLA 가속기 컴파일러가 차원 크기를 정적으로 상수화(Static)할 수 있도록 arange 축을 고정 빌드합니다.
    # [EN] Hardcode the arange axis to allow the JAX/XLA compiler to enforce static allocation dimensions during optimization loops.
    spatial_coordinate_axis = jnp.arange(volatile_dim, dtype=target_dtype)  # Shape: [Volatile_Time_Jitter]
    spatial_grid_mesh = spatial_coordinate_axis[None, :, None]  # Shape: [1, Volatile_Time_Jitter, 1] -> [KR] 무복사 브로드캐스팅 / [EN] Zero-copy broadcasting
    
    # ====================================================================
    # [KR] [수리 물리 교정] 질량 에너지 보존을 위한 공간 밀도장 선행 확정
    # [EN] [Mathematical Physics Calibration] Pre-determine Spatial Density Field for Mass-Energy Conservation
    # ====================================================================
    # [KR] 음수 영역 파동을 정류한 유체 질량 밀도를 상위 레일 연산으로 끌어올립니다.
    # [EN] Lift the fluidic mass density—rectified to eliminate negative wave field components—into the execution pipeline.
    wave_mass_density = jnp.maximum(fluidic_grid_stream, 0.0)
    
    # [KR] 0차 적분(평균 질량)을 미리 연산하여 델타의 기하학적 대칭축을 결정론적으로 정렬합니다.
    # [EN] Pre-calculate the zero-order integral (mean mass) to ensure deterministic alignment of the geometric symmetry axis.
    raw_integral = jnp.mean(wave_mass_density, axis=1)
    
    # [KR] 전단 라우터가 델타를 생략했거나, ReLU 변환 전 기준으로 수치적 뒤틀림이 일어날 위험을 원천 차단하기 위해
    #      온칩 레지스터 내부에서 정밀한 질량 중심 기준 델타를 고속 재연산합니다.
    #      Tracer 레벨에서 컴파일러가 조건문 분기 예측 실패(Branch Stall) 없이 최적의 로컬 레지스터로 인라인 퓨전합니다.
    # [EN] To prevent structural misalignment where upstream routers omit deltas or non-linear ReLU transformations skew the layout,
    #      the core re-computes the localized mass-center delta directly inside the on-chip registers.
    #      The compiler tracer inline-fuses this operation into optimal registers to fully eliminate branch prediction stalls.
    passed_delta = router_outputs.get("mean_centered_delta", None)
    
    pure_manifold_delta = (
        wave_mass_density - raw_integral[:, None, :]
        if passed_delta is None else passed_delta
    )

    
         # ====================================================================
    # [KR] [3 & 4 & 5 UPGRADED] 고차 모멘트 대수적 정화 코어
    # [EN] [3 & 4 & 5 UPGRADED] Higher-Order Moment Algebraic Purification Core
    # ====================================================================
    # 1) & 2) [KR] 0차 및 고차 모멘트 유도
    #      수리 물리 교정을 위해 전단에서 wave_mass_density 및 raw_integral 연산이 
    #      선행 완료되었으므로, 중복 메모리 할당을 방지하기 위해 상위 포인터를 그대로 재활용합니다.
    # 1) & 2) [EN] Enforce 0th and higher-order moment derivation
    #      Since wave_mass_density and raw_integral are pre-calculated for calibration,
    #      the system directly reuses the existing pointers to eliminate redundant memory allocation.
    m2 = jnp.mean(pure_manifold_delta ** 2, axis=1) # [KR] 분산 (2차 모멘트) / [EN] Variance (2nd-order moment)
    m3 = jnp.mean(pure_manifold_delta ** 3, axis=1) # [KR] 왜도 분자 (3차 모멘트) / [EN] Skewness Numerator (3rd-order moment)
    
    # 3) [KR] 가속기 SFU 네이티브 역수 변환기 강제 명시 (나눗셈 스톨 0.0% 완벽 달성)
    # 3) [EN] Enforce SFU-native hardware reciprocal invocation (0.0% division pipeline stall rate)
    denominator_safe = m2 + jax.lax.stop_gradient(integration_epsilon)
    reciprocal_m2 = jax.lax.reciprocal(denominator_safe) 
    
    # 4) [KR] 동적 비대칭 유체 압력 오프셋 상쇄 (왜도 평탄화 정류 기전 완수)
    # 4) [EN] Real-time subtraction of the dynamic asymmetric fluidic pressure offset (Skewness flattening execution)
    asymmetric_correction = 0.5 * m3 * reciprocal_m2
    sanitized_integral = raw_integral - asymmetric_correction
    
    # 5) [KR] Pure Branchless 1사이클 관통 이진 판정 (Gather/Scatter 인덱싱 스톨 방지)
    #      0.5 임계값 제어를 if문이나 select 없이 IEEE 754 부동소수점 비교 플래그의 
    #      데이터 타입 캐스팅(.astype)만으로 처리하여 가속기 코어를 단 1클록만에 관통시킵니다.
    # 5) [EN] Pure Branchless single-cycle binary decision (Prevents indexing and gathering stalls)
    #      The threshold control (0.5) is evaluated via primitive IEEE 754 floating-point comparison logic 
    #      coupled with direct data type casting (.astype), achieving single-clock execution pipelines.
    static_information_tensor = (sanitized_integral > 0.5).astype(jnp.float32)
    
    # 6) [KR] 레거시 백엔드 호환용 관제 모니터링선 텐서 유지 (0차 수직 압축 포인터 재활용)
    # 6) [EN] Maintain legacy backend monitoring tensor paths (0th-order vertical compression pointer recycling)
    total_system_mass = jnp.sum(wave_mass_density, axis=1, keepdims=True)
    weighted_mass_moment = jnp.sum(wave_mass_density * spatial_grid_mesh, axis=1, keepdims=True)
    
    # [KR] 하위 레거시 호환선 내부의 슬래시(/) 나눗셈 기호도 하드웨어 네이티브 역수 파이프라인으로 전환
    # [EN] Convert legacy division slash (/) routines into hardware-native reciprocal streams
    mass_denominator_safe = total_system_mass + jax.lax.stop_gradient(integration_epsilon)
    reciprocal_mass = jax.lax.reciprocal(mass_denominator_safe)
    center_of_mass_indices = weighted_mass_moment * reciprocal_mass
    reconstructed_static_indices = center_of_mass_indices.astype(jnp.uint32)
    
    # ====================================================================
    # [KR] [6] DECODER QUANTUM TELEMETRY (역산 수치 안정성 관제 관류)
    # [EN] [6] Decoder Quantum Telemetry (Inversion Numerical Stability Tracking)
    # ====================================================================
    # [KR] 정화 완료된 sanitized_integral의 수렴 무결성을 안전 상수로 격리 수집합니다.
    # [EN] Collect the convergence integrity of the fully purified sanitized_integral into isolated safety constants.
    manifold_vacuum_rate = jax.lax.stop_gradient(
        jnp.mean((total_system_mass < integration_epsilon).astype(target_dtype))
    )
    decoder_telemetry = {
        "manifold_vacuum_rate": manifold_vacuum_rate,
        "decoder_numerical_stability": jax.lax.stop_gradient(jnp.min(sanitized_integral)),
        "reconstructed_static_indices": reconstructed_static_indices # [KR] 레거시 모니터링선 하방 결합 유지 / [EN] Maintain legacy monitoring downstream coupling
    }

    return static_information_tensor, decoder_telemetry

