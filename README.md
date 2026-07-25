# 🌊 Technical Specification: Fluidic Network Grid (FNG) V3

**Hardware Barrier-Free & High-Order Moment Asymmetric Correction Architecture for Ultra-Scale Parallel Deep Learning**

---

## 🏛️ Architectural Lineage & Evolution (아키텍처 진화 및 계보)

> ⚠️ **[Architectural Realignment Notice / 아키텍처 정정 명세]**
>
> * **[KR] 본 저장소의 아키텍처 전환 고지:** 
>   기존 `fluidic_mockup/` 폴더 내의 모듈들은 분산 AI 인프라의 전송 지터 및 난류 현상을 거시적 유체 제어 수식으로 시각화한 개념 실증(Proof of Concept) 모델이었습니다. 본 프로젝트는 해당 목업이 제안했던 JAX 컴파일러 메시 분할 레이아웃을 계승하되, 수치 해석적 왜곡을 유발하던 이진 부호 변환 루틴을 제거하고 `bfloat16 / float32` 고정밀 연속 소수점 데이터 레일 위에서 정상 작동하는 **실전형 가속 패키지(Production Core)로의 아키텍처 피벗을 완료**했습니다.
>
> * **[EN] Important Infrastructure Realignment Notice:** 
>   The fluidic control formulations inside `/fluidic_mockup` have been completely re-engineered into a production-grade acceleration suite. By eliminating the previous binary-clipping data destruction routines, this architecture serves as a production-ready framework interlocked natively with high-density LLM training and inference rails.

### 🧩 개념 목업(Mock-up)에서 프로덕션(Production Core)으로 계승된 기술적 핵심 자산
본 패키지(`production_core/`)는 선행 기획 및 목업 자산들로부터 아래의 시스템 엔지니어링 설계 철학을 유기적으로 상속받아 세공되었습니다:

1. **Neumann Boundary 경계면 패딩 제약 기전 (`core_smoother_xla.py`)**
   - 분산 환경 내 데이터 단절 및 하드웨어 불연속면에서 발생할 수 있는 수치적 발산(Divergence)을 완화하기 위해, 기존 기획서에서 명세했던 **"Neumann Boundary 조건(경계면 기울기 0 고정)" 기반의 엣지 패딩(Padding) 제어 기믹을 계승**하여 구현했습니다.
2. **SFU 하드웨어 네이티브 지수 회로 결착 기전 (`elastic_governor.py`)**
   - 비선형 시그모이드 가변 점성 스케일링 수식 연산 시 발생하는 나눗셈 명령어 오버헤드를 줄이기 위해, 가속기 내부 **SFU(Special Function Unit) 명령어 단으로 직접 융합 컴파일(Inline Fusion)**되도록 바인딩한 하드웨어 친화적 수식 전개 방식을 유지했습니다.
3. **`shard_map` 기반의 정적 다차원 메시 격자 사양 (`async_scheduler.py`)**
   - 분산 클러스터 토폴로지의 디바이스 축만 기하학적으로 분할(Sharding)하고 시간축과 특징 차원을 하드웨어 레지스터에 정적으로 배치하여, **추가적인 VRAM 메모리 할당 오버헤드를 최소화하고 데이터 스트림을 관류**시키려 했던 오케스트레이션 설계 사양을 계승했습니다.


---

## 🎨 Dual-Pathway Systems Topology (이원화 아키텍처 구조)

본 레포지토리는 수리물리학적 상상력을 분산 AI 하드웨어 가속기 제어에 투사한 엔진입니다. 초기 방향성 제시용 모델과 그로부터 발전된 개발의 연속성과 학술적 자산을 보존하기 위해 시스템을 두 가지 경로로 나누어 오픈형으로 공개합니다.

```directory
Fluidic_Network_Grid/
├── 🌊 fluidic_mockup/          # [개념 실증 목업] 유체역학 방정식 기반 시뮬레이터 원형 아카이브
└── 🚀 production_core/         # [실 상용화 코어] bfloat16 소수점을 사수하는 실제 고속 가속 엔진 패키지
```

---
## 📊 Quantitative Architectural Realignment (정량적 명세 비교)

| 하드웨어 / 수학 레이어 | 전작의 수리물리 목업 (`/fluidic_mockup`) | 본 프로젝트의 실전 상용화 기술 (`/production_core`) |
| :--- | :--- | :--- |
| **최종 기술 목적 (Goal)** | 유체 수식을 통한 가상 공간 내 오차 보정 시뮬레이션 | **대규모 분산 SGD 학습 중 그라디언트 폭발 및 NaN 발산 안정화** |
| **데이터 정밀도 (Precision)**| `> 0.5` 임계값 필터 적용으로 수치 해석적 단선 유발 | **`bfloat16 / float32` 연속적 소수점 그라디언트 정밀도 보존** |
| **수리물리 도메인 정의** | 데이터 손실(Blackout) 시 결측치를 대수적으로 합성 보간 | **고주파 수치 변위를 Burgers' 점성 제어로 감쇠하는 정류 필터** |
| **통신 스케줄링 기믹** | 추상적 우회 가설 제시 (동기식 NCCL `psum` 배리어에 의존) | **XLA 백엔드 단에서의 정적 연산-통신(All-Reduce) 비동기 중첩** |
| **무선 결함 허용 (Fault)** | 가상 대기 주소 풀(`cold_standby_pool`) 기반의 상태 복구 | **`stop_gradient` 캐시 락킹 및 내재적 항상성 기반 에지 Elastic Rescue** |
| **출력 차원 명세 (Specs)** | 지터 축 가변성 유실에 의한 3D 파티셔닝 `P(M, N, None)` | **지터 차원을 보존하여 Llama SDPA 회로와 직결되는 4D `P(N, M, N, N)`** |

---

## 📐 3-Tier Production Operational Pipeline (프로덕션 코어 명세)

본 프로덕션 코어(`production_core/`) 패키지는 가속기 내부 온칩 메모리(SRAM) 효율을 개선하고 프레임워크 지터의 개입을 방지하기 위해 3개의 독립적인 최적화 레이어로 설계되었습니다.

### 🌊 Layer 1: 점성 기반 그라디언트 난류 정류 (`core_smoother_xla.py`)
- **기능:** Burgers' 방정식의 소산 항($+\sigma \frac{\partial^{2} \mathbf{\Phi}}{\partial x^{2}}$)을 역이용하여, 분산 SGD 학습 중 터져 나오는 고주파 수치 요동을 점성 소산 기전으로 감쇠합니다.
- **수직 계통:** Neumann 경계 조건(경계면 기울기 0 고정)을 인플레이스로 반영하며, 부호 반올림에 의한 왜곡 없이 `bfloat16 / float32` 고정밀 연속 소수점 그라디언트 다양체의 무결성을 유지합니다.

### ⚡ Layer 2: 무분기 가속기 방화벽 및 미분 선로 보호 (`math_guardrails.py`)
- **기능:** 연산 과정에서 유발되는 `NaN / INF` 예외 및 임계치 초과 스파이크를 조건부 분기문(JMP) 없이 하드웨어 MUX 선택자(`selp.f32`)와 1:1 결착하여 원자적으로 플러시합니다.
- **수직 계통:** 값을 상수로 평탄화하여 미분 계수를 0으로 만드는 고질적 단점을 우회하고, 임계면 바깥 영역에 미세 기울기(`Leaky Slope`)를 합성하여 오차 역전파 미분 사슬을 안정적으로 사수합니다.

### 🎛️ Layer 3: 하이브리드 토폴로지 오케스트레이션 및 동적 핫스왑
- **유선 데이터센터 패스 (`async_scheduler.py`):** JAX `shard_map` 기반의 정적 4차원 차원 고정 매핑을 구동합니다. 수리물리 정류 연산 파이프라인의 실행 타이밍과 백그라운드 레지스터 단의 NCCL All-Reduce 통신(`psum`)을 동시에 오버랩하여 전송 레이턴시를 은닉(Latency Hiding)합니다.
- **무선 에지 가드 패스 (`elastic_governor.py`):** 불필요한 가상 버퍼 및 메모리 참조 오버헤드를 최소화하고, `jax.lax.scan` 피드백 루프 내에서 시스템 스스로 직전 스텝까지 입증해 낸 청정 텐서에 `stop_gradient` 락을 발동합니다. 패킷 드롭률 85% 이상의 극한 무선 지터 환경에서도 전역 가중치 오염 없이 파이프라인을 유지합니다.

---


## ⚡ Integrated Benchmark Test System (`test_production_pipeline.py`)

본 프로젝트는 가상 분산 환경과 하드웨어 클러스터 모두에 범용 대응하는 단독 가동형 검증 시스템을 저장소 루트에 포함하고 있습니다. `production_core` 전체 모듈의 수치적 정합성과 유·무선 실행 선로의 핫스왑 구동 상태를 실시간으로 전수 오디트(Audit)합니다.

### 🔬 2-Stage Stress Fault Injection (결함 주입 테스트 환경)

1. **🏢 Stage 1: Wired Datacenter Jitter Track**
   - **인프라 모사:** NVLink 인프라 내부의 고속 분산 SGD 전송 패스를 시뮬레이션합니다.
   - **난류 주입:** 미세 패킷 전송 시차 및 인프라 지터 요동(10% 무작위 지터 마스크)을 상시 가산하여 `async_scheduler.py` 커널의 온칩 연산-통신 완전 중첩(Overlapping) 레이턴시 은닉 효율을 프로파일링합니다.
   
2. **📡 Stage 2: Wireless Edge Extreme Blackout Track**
   - **인프라 모사:** 대규모 패킷 파열 및 불안정한 통신 단선(Disconnection) 인프라를 모사합니다.
   - **난류 주입:** 4차원 시퀀스 데이터 스트림 전면에 **88% 이상의 대규모 패킷 드롭 및 전송 단선(Blackout) 결함**을 주입합니다. 이를 통해 외부 데이터 단류 상태에서 `elastic_governor.py`가 스스로 무결함을 증명해낸 직전 Carry 상수를 `stop_gradient`로 안전하게 락인(Lock-in)하여 전역 가중치 단으로의 `NaN` 오염 확산을 차단해 내는지 계측합니다.


### 💻 Execution & Verification Command (구동 명령어)

레포지토리 루트 디렉토리에서 아래 명령어를 가동하여 인프라 커널의 수치 안정화 기전을 검증할 수 있습니다:
```bash
python production_core/test_production_pipeline.py
```

### 📺 Expected Audit Console Output (합격 로그 양식)

무분기 마스킹 연산이 정상 결착되었을 때 출력되는 표준 콘솔 로그 양식입니다:

```text
================================================================================
🌊 [FNG V3 PRODUCTION CORE - INTEGRATED BENCHMARK SYSTEM START]
================================================================================
✅ [WIRED_DATACENTER SUCCESS]: Burgers' Damping complete.
✅ [WIRELESS_EDGE SUCCESS]: 88% Blackout bypassed via internal static manifold.
🏆 [FINAL CONCLUSION]: ALL FNG V3 PRODUCTION KERNELS PASSED!
================================================================================
```

---

## 📜 License & Copyleft Notice 

본 프로젝트는 **Apache License 2.0** 오픈소스 라이선스 하에 배포됩니다.

- **상용 및 학술적 이용 권리:** 본 저장소의 핵심 아키텍처 기전(`shard_map` 4D 파티셔닝 명세, Burgers' 점성 기반 그라디언트 평탄화, Leaky Slope 기반 수치 안정화 레이어)은 오픈소스 생태계의 인프라 설계자, 연구원, 기업 클러스터가 자유롭게 포크(Fork)하여 상용 시스템 및 학술 연구에 인용·변형·탑재할 수 있습니다.
- **기여 및 계보 존중:** 본 엔진을 기반으로 추가적인 리팩토링이나 프레임워크 플러그인을 확장 배포하실 경우, 원작자(`PJHkorea`)의 기하학적 3축 오케스트레이션 설계 청사진과 수리물리학적 계보(Lineage)를 README 및 소스코드 헤더 명세에 명시적으로 인용해 주셔야 합니다.


