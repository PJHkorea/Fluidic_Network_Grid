import jax
import jax.numpy as jnp
from functools import partial


from production_core.async_scheduler import compile_asynchronous_overlapping_pipeline
from production_core.elastic_governor import compile_wireless_elastic_governor

class FngInterleavedLlamaAttention(object):
    """
    [FNG V3 PRODUCTION CORE - HIGH-LEVEL NEURAL CO-DESIGN ADAPTER]
    
    기존 fluidic_mockup의 추상적인 설정을 완전히 청산하고,
    실제 Llama/DeepSeek 스타일의 분산 컨텍스트 병렬화(Context Parallelism) 환경에서
    유·무선 하이브리드 인프라 제어 및 핫스왑을 집행하는 실전형 트랜스포머 어댑터입니다.
    """
    def __init__(self, devices_mesh, mesh_axis_name="fluidic_mesh"):
        self.devices_mesh = devices_mesh
        self.mesh_axis_name = mesh_axis_name
        
        # 🏢 1) [유선 데이터센터] 컴파일 타임 퓨전 마스터 레이아웃 사전 빌드
        # JAX/XLA 컴파일러 내부 그래프 메모리에 비동기 겹치기 팩토리를 미리 동결 바인딩합니다.
        self.v1_async_scheduler = compile_asynchronous_overlapping_pipeline(
            devices_mesh=self.devices_mesh,
            mesh_axis_name=self.mesh_axis_name
        )
        
        # 📡 2) [무선 에지 가드] 컴파일 타임 항상성 제어 레이아웃 사전 빌드 (추가)
        # jax.lax.scan 기반의 비선형 가변 점성 및 stop_gradient 가드 팩토리를 사전 동결합니다.
        self.v2_elastic_governor = compile_wireless_elastic_governor(
            devices_mesh=self.devices_mesh,
            mesh_axis_name=self.mesh_axis_name
        )


        def __call__(self, local_q, local_k, local_v, pollution_mask, 
                 viscosity_sigma=3.125e-5, integration_epsilon=1e-6, 
                 deploy_env="WIRED_DATACENTER", initial_state=None, current_drop_rate=0.0):
        """
        트랜스포머 포워드 패스에서 실시간으로 호출되는 하이브리드 어텐션 퓨전 엔트리 포인트.
        
        [Manifold Dimension Alignment Specification / 차원 일치 명세]
        - local_q (Query): [Batch, Head_Dim, Feature_Dim] (표준 3D Llama 레이아웃)
        - local_k / local_v (Key/Value): [Batch, Volatile_Time_Jitter_Dim, Feature_Dim] (유선 V1 모드 시 3D)
                                         혹은 [Time_Steps, Batch, Volatile_Time_Jitter_Dim, Feature_Dim] (무선 V2 모드 시 4D 시퀀스)
        """
        target_dtype = local_q.dtype
        sigma_tensor = jnp.array(viscosity_sigma, dtype=target_dtype)
        epsilon_tensor = jnp.array(integration_epsilon, dtype=target_dtype)
        
        # --------------------------------------------------------------------------
        # ⚡ STEP 1: 유무선 환경(deploy_env)에 따른 고속 인프라 선로 핫스왑 (Hot-swap)
        # --------------------------------------------------------------------------
        if deploy_env == "WIRED_DATACENTER":
            # 🏢 1-1) 유선 데이터센터: 고정 점성 기반 연산-통신 오버랩 파이프라인 가동 (V1 패스)
            purified_k = self.v1_async_scheduler(local_k, pollution_mask, sigma_tensor, epsilon_tensor)
            purified_v = self.v1_async_scheduler(local_v, pollution_mask, sigma_tensor, epsilon_tensor)
            
        elif deploy_env == "WIRELESS_EDGE":
            # 📡 1-2) 무선 에지: 시그모이드 가변 점성 및 jax.lax.scan 기반 항상성 가드 가동 (V2 패스)
            if initial_state is None:
                # 초기 캐리 상태 구성 [초기 점성, 초기 청정 백업 레일]
                initial_state = (sigma_tensor, jnp.zeros_like(local_k[0] if local_k.ndim == 4 else local_k))
                
            # [교정 완료]: 하방으로 전사할 튜플 스펙 동결 및 가속기 scan 실행
            # elastic_governor의 4D out_specs 교정 사양에 맞춰 출력 스트림과 텔레메트리를 완벽히 분해 분리합니다.
            k_seq, _ = self.v2_elastic_governor(local_k, initial_state)
            v_seq, _ = self.v2_elastic_governor(local_v, initial_state)
            
            # [★CRITICAL PROD REFACTORING★] 
            # 4차원 시퀀스 타임라인 중 소프트맥스 연산에 직결 가능한 최신 타임스텝의 단면[-1 인덱스]을 
            # 무분기로 안전하게 적출하여 3차원 표준 Llama 다양체 평면으로 강제 정렬합니다.
            purified_k = k_seq[-1]
            purified_v = v_seq[-1]
            
        else:
            raise ValueError(f"❌ [FNG ERROR]: Invalid deployment environment configuration: {deploy_env}")

        # --------------------------------------------------------------------------
        # 📐 STEP 2: Non-Blocking Tensor Geometric Alignment (기하학적 차원 정렬 & SDPA)
        # --------------------------------------------------------------------------
        # 표준 Llama Scaled Dot-Product Attention 수식 회로 직통 결착
        # 하드웨어 스토리지 보폭 일치를 위해 맨 마지막 특징 차원(Feature_Dim) 적출
        head_dim = local_q.shape[-1] 
        scaling_factor = jnp.sqrt(jnp.array(head_dim, dtype=target_dtype))
        
        # 🧮 Score = (Q ✕ K^T) / sqrt(d_k)
        # 축 정렬 원자적 행렬곱 연산 (Batch Matrix Multiplication)
        # 3차원으로 완전 정류된 다양체 평면을 기반으로 transpose 및 matmul을 안전하게 집행하여 차원 크래시를 영구 박멸합니다.
        attention_scores = jnp.matmul(local_q, jnp.transpose(purified_k, (0, 2, 1))) / scaling_factor
        attention_weights = jax.nn.softmax(attention_scores, axis=-1)
        
        # 🧮 Context Vector = Softmax(Score) ✕ V
        # 최종 도출되는 문맥 벡터는 복잡한 조건부 분기문 정체 없이 [Batch, Head_Dim, Feature_Dim]의 청정 뷰로 수렴
        context_vector = jnp.matmul(attention_weights, purified_v)
        
        return context_vector
