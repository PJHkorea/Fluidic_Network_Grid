# 📡 FNG V3 Production Core: Developer Reference

본 디렉토리는 분산 대형 언어 모델(LLM)의 학습 및 추론 파이프라인 최적화를 위한 **JAX/XLA 기반 하이브리드 인프라 커널 패키지**입니다. 수리물리학적 연속성 가설을 분산 가속기 제어 평면에 이식하여 수치 안정성과 통신 오버랩 효율을 개선합니다.

## 📂 Core Module Architecture
```directory
production_core/
├── __init__.py                     # 정적 네임스페이스 격리 및 런타임 룩업 오버헤드 최소화
├── core_smoother_xla.py            # Burgers' 방정식 소산 기전을 역이용한 그라디언트 난류 평탄화
├── math_guardrails.py              # Leaky Slope 기반의 미분 연속성 보존형 NaN/INF 방화벽
├── async_scheduler.py              # [🏢 유선] shard_map 기반 연산-통신(All-Reduce) 정적 중첩 스케줄러
├── elastic_governor.py             # [📡 무선] jax.lax.scan 피드백 루프 기반 고결함 에지 복원 거버너
├── pytorch_mega_adapter.py         # PyTorch-to-JAX 간 DLPack Zero-Copy 공유 메모리 어댑터
├── transformer_fused.py            # 유무선 분기 및 표준 Llama 3D SDPA 직결 통합 어댑터
├── test_production_pipeline.py     # 가상 분산 메시 기반의 유무선 핫스왑 수치 무결성 검증 런처
└── test_megatron_speedup_report.py # Llama-3-70B/DeepSeek-V3 규격 가속기 TFLOPS 프로파일러 런처
```


## 🛠️ Key API Interface (`transformer_fused.py`)
`FngInterleavedLlamaAttention`은 호스트 단의 추상화 누수(Abstract Leak)를 차단하기 위해, 초기화 시점에 유무선 가속 팩토리 커널을 사전 빌드(Pre-compile)하여 런타임 추적(Tracer) 병목을 방지합니다.

```python
from production_core.transformer_fused import FngInterleavedLlamaAttention

# 🏗️ 분산 토폴로지 메시 바인딩 및 어댑터 초기화
fng_gate = FngInterleavedLlamaAttention(
    devices_mesh=devices_mesh, 
    mesh_axis_name="fluidic_mesh"
)

# ⚡ 순방향 패스 컴파일러 기반 핫스왑(Hot-swap) 실행
# deploy_env 플래그에 따라 하드웨어 실행 경로가 컴파일러 단에서 분기됩니다.
context_vector = fng_gate(
    local_q=local_q,               # Query [Batch, Head_Dim, Feature_Dim] (표준 3D)
    local_k=local_k,               # Key   [Batch, Jitter_Dim, Feature_Dim] (V1: 3D / V2: 4D)
    local_v=local_v,               # Value [Batch, Jitter_Dim, Feature_Dim] (V1: 3D / V2: 4D)
    deploy_env="WIRED_DATACENTER", # "WIRED_DATACENTER" 또는 "WIRELESS_EDGE"
    current_drop_rate=0.0          # 고결함(무선 에지) 환경 시 텔레메트리 유실률 지표
)
```

## 🏢 분산 배치 명세 (`async_scheduler.py`)
연속적인 소수점 상태의 지터 차원을 온전히 보존하기 위해 `P(None, mesh_axis_name, None, None)` 형태의 4D `PartitionSpec`을 채택합니다. 이를 통해 추가적인 메모리 복사 오버헤드 없이 글로벌 텐서 스트림을 분산 가속기 격자에 정적으로 매핑(Sharding)합니다.

---

## ⛓️ Downstream Numerical Pipeline Chain (수직 데이터 흐름)

본 패키지의 연산 체인은 불필요한 메모리 할당 및 호스트-디바이스 간 분기문 간섭을 최소화하기 위해 단일 정적 XLA 그래픽 트리로 융합되어 실행됩니다.

1. **`core_smoother_xla.py` [수치 정류 레이어]**
   - 4차원 그라디언트 다양체의 요동을 Burgers' 방정식의 소산 항으로 완화하고, 2차·3차 모멘트 역산을 통한 대수적 비대칭(Skewness) 평탄화를 수행합니다.
2. **`math_guardrails.py` [수치 안정화 레이어]**
   - 이전 정류 단을 통과한 텐서 내의 `NaN / INF` 부호를 포획하여 지정된 베이스라인(`0.0f`)으로 안정화하되, 임계값 초과 영역에 `leaky_slope` 미세 기울기를 합성하여 역전파 미분 사슬의 단선을 방지합니다.
3. **핫스왑 인터록 인터페이스 (`async_scheduler.py` / `elastic_governor.py`)**
   - **유선 모드 (`WIRED_DATACENTER`):** 정류 연산 파이프라인의 실행 타이밍과 `jax.lax.psum` All-Reduce 프리미티브를 백그라운드에서 동시 중첩(Overlap)하여 분산 통신 레이턴시를 은닉합니다.
   - **무선 모드 (`WIRELESS_EDGE`):** 패킷 드롭률이 85%를 초과하는 극단적 불안정 환경 진입 시, `jax.lax.stop_gradient`를 활성화하여 직전 타임스텝의 정상 Carry 상태를 동결·참조함으로써 전역 가중치의 오염 확산을 차단합니다.

---

## ⚠️ Accelerator Compiler Considerations (가속기 컴파일 주의사항)

본 코어를 타 인프라 및 트랜스포머 레이어에 이식할 시, XLA 백엔드의 컴파일 최적화 안정성을 위해 다음 규칙을 엄격히 준수해야 합니다:

1. **정적 디바이스 메시 구조 동결 (`devices_mesh`)**
   - JAX/XLA 컴파일러의 전역 팩토리 추적 특성상, 런타임 중에 디바이스 메시의 축 개수나 가속기 토폴로지 레이아웃이 동적으로 변경되면 트레이서 오류가 발생하거나 대규모 재컴파일(Tracer Re-trace) 스톨이 유발됩니다. 반드시 인스턴스 초기화(`__init__`) 시점에 정적 Mesh 구조를 확정 및 고정하여 주입하십시오.
2. **4차원 다양체 데이터 규격 일치 (`PartitionSpec`)**
   - 패키지 내의 `async_scheduler`와 `elastic_governor` 커널은 상호 선로 전환의 무결성을 위해 `P(None, mesh_axis_name, None, None)` 기반의 4D 파티셔닝 명세를 완벽히 공유합니다. 외부 프레임워크나 KV 캐시 레이어에서 특정 차원을 임의로 축소(Squeeze/Unsqueeze)하여 통과시킬 경우 축 비대칭 에러(`ValueError: axes do not match`)가 발생하므로 입출력 텐서의 랭크 사양을 상시 일치시키십시오.

---

## 🛠️ PyTorch Framework Integration (`pytorch_mega_adapter.py`)
Megatron-LM 및 DeepSeek-V3급 컨텍스트 병렬화(Context Parallelism) 선로와의 유기적인 상호 연동을 위해 PyTorch 모듈 래퍼를 제공합니다. `FngPyTorchMegaAdapter`는 DLPack 공유 메모리 인터페이스를 경유하여, PyTorch 텐서와 JAX/XLA 커널 컨텍스트 간의 0바이트 무복사(Zero-Copy) 포인터 뷰(View) 매핑을 수행합니다.

```python
from production_core.pytorch_mega_adapter import FngPyTorchMegaAdapter

# 🏗️ PyTorch 분산 그래프 내 모듈 초기화
fng_torch_adapter = FngPyTorchMegaAdapter(devices_mesh=devices_mesh, mesh_axis_name="fluidic_mesh")

# ⚡ PyTorch Forward Attention Block 내 무복사 관류 집행
# 호스트-디바이스 간 컨텍스트 동기화 최소화를 위해 XLA 내부 하드웨어 락 배리어가 연동됩니다.
output_context = fng_torch_adapter(
    pytorch_q=q_tensor,             # PyTorch Tensor [Batch, Seq_Len, Hidden_Dim]
    pytorch_k=k_tensor,             # V1 모드 시 3D / V2 모드 시 4D Sequence Tensor
    pytorch_v=v_tensor,             # V1 모드 시 3D / V2 모드 시 4D Sequence Tensor
    pytorch_pollution_mask=mask,    # PyTorch 가속기 메모리 상의 지터 노이즈 마스크
    deploy_env="WIRED_DATACENTER"
)
```

## 📊 Megatron Speedup Profiler (`test_megatron_speedup_report.py`)
초대형 모델 병렬화 사양 하에서 수치 정정 성능 및 통신 효율 향상률을 ns 단위로 정산하는 단독 실행형 가속기 프로파일러 런처입니다.

- **실리콘 클록 동기화 포획:** Python 호스트 단의 가비지 컬렉터(GC) 레이턴시 오염을 배제하기 위해 `torch.cuda.Event` 하드웨어 인터럽트를 직접 제어하여 최소 레이턴시(Min Latency)를 징수합니다.
- **TFLOPS 및 은닉율 정량 계산:** 수립된 하드웨어 타이머와 Llama 표준 대수 연산량 수식(Flops)을 대조하여 순수 가속기 TFLOPS 처리량 및 기존 NCCL 동기 장벽(Barrier) 대비 오버랩 은닉 효율(Δ %)을 최종 리포팅 대시보드로 사출합니다.

---
## 📜 Copyright & Copyleft
- **Apache License 2.0**
- 본 코어의 수리물리 필터 및 4D Shard-Map 오케스트레이션 설계 원형을 포크 및 전사하여 활용하실 경우, 원작자(**PJHkorea**)의 엔지니어링 계보(Lineage)를 코드 소스 헤더 및 레퍼런스에 명시해 주셔야 합니다.
