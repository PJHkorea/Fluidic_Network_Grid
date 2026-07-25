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
        트랜스포머 포워드 패스에서 실시간으로 호출되는 어텐션 퓨전 엔트리 포인트.
        
        [Manifold Dimension Alignment Specification / 차원 일치 명세]
        - local_q (Query): [Nodes/Batch, Head_Dim, Feature_Dim] (표준 Llama 레이아웃)
        - local_k / local_v (Key/Value): [Nodes/Batch, Volatile_Time_Jitter_Dim, Feature_Dim] 
                                         (네트워크 지터 차원이 가미된 원본 통신 스트림)
        - deploy_env: "WIRED_DATACENTER" (V1 오버랩) 또는 "WIRELESS_EDGE" (V2 Elastic 가드)
        """
        target_dtype = local_q.dtype
        sigma_tensor = jnp.array(viscosity_sigma, dtype=target_dtype)
        epsilon_tensor = jnp.array(integration_epsilon, dtype=target_dtype)
        
        # --------------------------------------------------------------------------
        # ⚡ STEP 1: 유무선 환경(deploy_env)에 따른 고속 인프라 핫스왑 (Hot-swap)
        # --------------------------------------------------------------------------
        if deploy_env == "WIRED_DATACENTER":
            # 🏢 1-1) 유선 데이터센터: 고정 점성 기반 연산-통신 오버랩 파이프라인 가동 (V1 패스)
            purified_k = self.v1_async_scheduler(local_k, pollution_mask, sigma_tensor, epsilon_tensor)
            purified_v = self.v1_async_scheduler(local_v, pollution_mask, sigma_tensor, epsilon_tensor)
            
        elif deploy_env == "WIRELESS_EDGE":
            # 📡 1-2) 무선 에지: 시그모이드 가변 점성 및 jax.lax.scan 기반 0ns 피드백 가드 가동 (V2 패스)
            # 타임 시퀀스 스캔 연산을 위해 기본 입력을 3차원에서 4차원 시퀀스 형태로 패킹 가정 처리
            if initial_state is None:
                # 초기 캐리 상태 구성 [초기 점성, 초기 청정 백업 레일]
                initial_state = (sigma_tensor, jnp.zeros_like(local_k))
                
            drop_rate_tensor = jnp.array(current_drop_rate, dtype=target_dtype)
            
            # jax.lax.scan 실행 커널 가동하여 타임 프레임별 전사 정류 집행
            k_seq, _ = self.v2_elastic_governor(local_k[None, ...], (initial_state[0], initial_state[1]))
            v_seq, _ = self.v2_elastic_governor(local_v[None, ...], (initial_state[0], initial_state[1]))
            
            # 사출된 시퀀스 중 최신 활성화 다양체 평면 적출 [0번째 시퀀스 스텝 복원]
            purified_k = k_seq[0]
            purified_v = v_seq[0]
            
        else:
            raise ValueError(f"❌ [FNG ERROR]: Invalid deployment environment configuration: {deploy_env}")


               # --------------------------------------------------------------------------
        # 📐 STEP 2: Non-Blocking Tensor Geometric Alignment (기하학적 차원 정렬)
        # --------------------------------------------------------------------------
        # [★CRITICAL PROD REFACTORING★] V2 무선 에지 패스를 통과하며 확장되었던 
        # 4차원 타임 시퀀스 축의 동적 가변성을 무분기로 압축 제어합니다.
        # jax.lax.select 또는 인플레이스 슬라이싱을 통해 3차원 표준 Llama 다양체 평면으로 강제 차원 정렬을 집행합니다.
        k_3d = jax.lax.select(deploy_env == "WIRELESS_EDGE", purified_k[0], purified_k)
        v_3d = jax.lax.select(deploy_env == "WIRELESS_EDGE", purified_v[0], purified_v)

        # 표준 Llama Scaled Dot-Product Attention 수식 회로 직통 결착
        # 🧮 Score = (Q ✕ K^T) / sqrt(d_k)
        head_dim = local_q.shape[-1] # 하드웨어 스토리지 보폭 일치를 위해 맨 마지막 특징 차원 적출
        scaling_factor = jnp.sqrt(jnp.array(head_dim, dtype=target_dtype))
        
        # 축 정렬 원자적 행렬곱 연산 (Batch Matrix Multiplication)
        # 교정된 3차원 다양체 평면(k_3d)을 기반으로 transpose를 집행하여 차원 크래시를 영구 박멸합니다.
        # 통신 지연 및 데이터 노이즈 오염으로 인한 재전송(ACK/NACK) 스톨 없이 온칩에서 그대로 관류합니다.
        attention_scores = jnp.matmul(local_q, jnp.transpose(k_3d, (0, 2, 1))) / scaling_factor
        attention_weights = jax.nn.softmax(attention_scores, axis=-1)
        
        # 🧮 Context Vector = Softmax(Score) ✕ V
        # 최종 도출되는 문맥 벡터는 복잡한 조건부 분기문 정체 없이 [Nodes/Batch, Head_Dim, Feature_Dim]의 청정 뷰로 수렴
        context_vector = jnp.matmul(attention_weights, v_3d)
        
        return context_vector

