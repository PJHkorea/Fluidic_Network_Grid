# 🚀 FNG V3 Production Core: Developer Reference

이 디렉토리는 분산 생성형 AI 모델의 학습 및 추론 파이프라인에 직접 직결되는 **상용 등급(Production-Grade) 가속 인프라 커널**을 포함합니다. JAX/XLA 최적화에 기반하여 작동합니다.

## 📂 Core Module Architecture
```directory
production_core/
├── __init__.py                # Namespace 동결 및 효율적 룩업 게이트
├── core_smoother_xla.py       # 그라디언트 난류 정류(Burgers' 점성 기반)
├── math_guardrails.py         # NaN/INF 원자적 플러시 방화벽
├── async_scheduler.py         # 🏢 유선: shard_map 기반 통신-연산 중첩 스케줄러
├── elastic_governor.py        # 📡 무선: jax.lax.scan 기반 85% 단선 항상성 복원
└── transformer_fused.py       # LLM SDPA 직결 통합 관제 어댑터
```

## 🛠️ Key API Interface (`transformer_fused.py`)
`FngInterleavedLlamaAttention`을 통해 유무선 가속 팩토리를 선차 컴파일하여 런타임 병목을 방지합니다.

```python
from production_core.transformer_fused import FngInterleavedLlamaAttention

# 🏗️ 인프라 초기화 및 바인딩
fng_gate = FngInterleavedLlamaAttention(devices_mesh=devices_mesh, mesh_axis_name="fluidic_mesh")

# ⚡ 실시간 순방향 패스 주사 (0ns 하드웨어 핫스왑)
context_vector = fng_gate(
    local_q=local_q,               # Query [Batch, Head_Dim, Feature_Dim]
    local_k=local_k,               # Key   [Batch, Jitter_Dim, Feature_Dim]
    local_v=local_v,               # Value [Batch, Jitter_Dim, Feature_Dim]
    deploy_env="WIRED_DATACENTER", # "WIRED_DATACENTER" 또는 "WIRELESS_EDGE"
    current_drop_rate=0.0          # 무선 모드 시 패킷 유실률 지표
)
```

## 🏢 유선 오케스트레이터 (`async_scheduler.py`)
데이터 연속 소수점 지터 차원을 보존하는 청정 4D `PartitionSpec`을 사용하여 0바이트 온칩 무복사 스트리밍을 수행합니다.

---

## ⛓️ Downstream Numerical Pipeline Chain (수직 계산 흐름)

본 패키지의 모든 연산은 데이터 복사(Zero-Copy) 및 분기문 정체 없이 하나의 정적 XLA 기계어 그래프로 동결되어 관류합니다.

1. **`core_smoother_xla.py` [물리 정류]**
   - 4차원 그라디언트 난류의 파동을 버거스 점성 소산 항으로 흡수하고 고차 모멘트 왜도(Skewness) 평탄화를 원자적으로 집행합니다.
2. **`math_guardrails.py` [수치 절연]**
   - 앞선 정류 단 바깥으로 유출된 NaN/INF 부호를 포획하여 0.0f 논리 레일로 플러시하되, `leaky_slope` 기울기를 강제 보존하여 미분 사슬을 절연 보호합니다.
3. **핫스왑 배포 인터록 (`async_scheduler.py` / `elastic_governor.py`)**
   - 유선 환경: 계산 가동 타이밍 뒤로 `jax.lax.psum` NCLL 통신 레이턴시를 100% 은닉합니다.
   - 무선 환경: 드롭률 85% 이상 시 `jax.lax.stop_gradient`로 직전 타임스텝의 청정 가중치를 락킹하여 가속기 폭사 생존을 지배합니다.

---

## ⚠️ Accelerator Compiler Considerations (가속기 구동 주의사항)

본 프로덕션 코어를 이식 및 빌드할 시 하부 가속기 분산 파티셔닝의 안정성을 위해 다음 규칙을 필히 엄수하십시오:

1. **정적 디바이스 메시 구조 동결 (`devices_mesh`)**
   - JAX/XLA 컴파일러 팩토리 패턴 특성상, 런타임 중에 디바이스 축의 토폴로지 개수나 레이아웃이 동적으로 변경되면 트레이서가 파열되며 대규모 재컴파일(Tracer Re-trace) 병목이 발생합니다. 반드시 초기화 단계(`__init__`)에서 고정된 Mesh 인스턴스를 주입하십시오.
2. **4차원 다양체 데이터 규격 일치 (`PartitionSpec`)**
   - `async_scheduler`와 `elastic_governor` 전 구간이 `P(None, mesh_axis_name, None, None)` 기반의 청정 4D 파티셔닝 정렬을 공유합니다. 상위 프레임워크나 KV 캐시 레이어에서 차원을 임의로 수축(Squeeze)하여 진입시키면 축 비대칭 에러가 발생하므로 규격 일치 상태를 상시 유지하십시오.

---

## 📜 Copyright & Copyleft
- **Apache License 2.0**
- 본 코어의 수리물리 필터 및 4D Shard-Map 오케스트레이션 설계 원형을 포크 및 전사하여 활용하실 경우, 원작자(**PJHkorea**)의 엔지니어링 계보(Lineage)를 코드 소스 헤더 및 레퍼런스에 명시해 주셔야 합니다.
