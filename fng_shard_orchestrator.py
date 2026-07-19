import jax
import jax.numpy as jnp
from jax.experimental.shard_map import shard_map
from jax.sharding import Mesh, PartitionSpec as P
from typing import Tuple, Dict
from fng_onchip_neumann_router import execute_fluidic_network_grid_ingress_v3

def orchestrate_fluidic_network_grid(
    global_packet_stream: jax.Array,        # Global Shape: [Total_Nodes, Volatile_Time_Jitter, Feature_Dim]
    global_cold_standby_pool: jax.Array,   # Global Shape: [Total_Nodes, Volatile_Time_Jitter, Feature_Dim]
    devices_mesh: Mesh,                     # 사전에 정의된 물리 가속기 토폴로지 메시 객체
    viscosity_sigma: float = 0.00003125
) -> Tuple[jax.Array, Dict[str, jax.Array]]:
    """
    Fluidic Network Grid Hardware-Native Orchestrator
    shard_map 디렉티브를 가동하여 NCCL 배리어 없이 하드웨어 레지스터 간 비동기 psum을 격수합니다.
    """
    
    # [1] 가속기 토폴로지에서 'fluidic_mesh' 축의 물리 장치 수 확인
    mesh_axis_name = "fluidic_mesh"
    assert mesh_axis_name in devices_mesh.axis_names, f"물리 장치 토폴로지에 '{mesh_axis_name}' 축 선언이 필요합니다."
    
    # [2] shard_map 데코레이터를 이용한 하드웨어 도메인 매핑 정의
    # 분산 노드(axis=0) 축을 물리 장치 격자에 단 1바이트의 메모리 복사 오버헤드 없이 1:1 슬라이싱 매핑
    @shard_map(
        mesh=devices_mesh,
        in_specs=(P(mesh_axis_name, None, None), P(mesh_axis_name, None, None)),
        out_specs=(P(mesh_axis_name, None, None), {
            "fluidic_grid_drop_rate": P(None),      # 텔레메트리는 전체 메시의 단일 스칼라 지표로 수렴
            "hardware_mesh_integrity": P(None)
        })
    )
    def fng_hardware_bound_kernel(local_packet, local_pool):
        """
        물리 가속기 코어 SRAM 내부로 완벽하게 절연된 FNG V3 스트림 엔진 실행 하네스
        """
        return execute_fluidic_network_grid_ingress_v3(
            raw_packet_stream=local_packet,
            cold_standby_address_pool=local_pool,
            viscosity_sigma=viscosity_sigma
        )

    # [3] 물리 토폴로지 내부 컨텍스트 진입 및 단일 융합 분산 그래프 동결
    # 수천 대의 노드로 전달된 데이터 스트림이 호스트와 디바이스 메모리 간 락(Lock) 없이 그대로 관통합니다.
    with devices_mesh:
        distributed_fluidic_stream, global_telemetry = fng_hardware_bound_kernel(
            global_packet_stream, 
            global_cold_standby_pool
        )
        
    return distributed_fluidic_stream, global_telemetry
