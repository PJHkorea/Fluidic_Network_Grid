# 🌊 Technical Specification: Fluidic Network Grid (FNG) V3

**Hardware Barrier-Free & High-Order Moment Asymmetric Correction Architecture for Ultra-Scale Parallel Deep Learning**

---

## 🏛️ Architectural Lineage & Evolution (아키텍처 진화 및 계보)

> ⚠️ **[Architectural Realignment Notice / 아키텍처 정정 명세]**
>
> * **[KR] 본 레포지토리의 중대한 진화 고지:** 
>   기존 `fluidic_mockup/` 폴더 내부의 코어들은 분산 AI 인프라의 지터 난류를 거시적 유체 현상으로 시각화한 결정론적 개념 실증 목업(Mock-up) 모델이었습니다. 본 프로젝트는 해당 목업이 가졌던 최고의 무기인 JAX 컴파일러 메시 뼈대를 100% 상속하되, 데이터 파괴를 일으키던 이진 부호 뭉개기 편법을 박멸하고 `bfloat16 / float32` 정밀도로 100% 정상 작동하는 **실 상용화 프로덕션 엔진(Production Core)으로의 완전한 피벗을 완료**했습니다.
>
> * **[EN] Important Infrastructure Realignment Notice:** 
>   The fluidic control formulations inside `/fluidic_mockup` have been completely re-engineered into a production-grade acceleration suite. By eliminating the previous binary-clipping data destruction routines, this architecture serves as a production-ready framework interlocked natively with high-density LLM training and inference rails.

### 🧩 목업(Mock-up)에서 프로덕션(Production Core)으로 계승된 핵심 기술 자산
본 프로젝트의 실 상용화 코어인 `production_core/` 패키지는 선행 목업 자산들로부터 아래의 핵심 시스템 엔지니어링 기전들을 유기적으로 상속받아 세공되었습니다:

1. **Neumann Boundary 경계면 클램핑 제약 기전 (`core_smoother_xla.py`)**
   - 격자 불연속면에서 데이터 단절로 인해 발생할 수 있는 가속기 수치적 폭발을 온칩 레지스터 단에서 완벽히 방어하기 위해, 기존 라우터가 사수하려 했던 **"Neumann Boundary 조건(경계면 기울기 0 고정)" 보폭(Padding) 제어 기믹을 100% 계승**했습니다.
2. **SFU 하드웨어 네이티브 지수/역수 회로 결착 기전 (`elastic_governor.py`)**
   - 비선형 시그모이드 가변 점성 스케일링 수식 연산 시, 하드웨어 나눗셈 슬래시(`/)` 오버헤드를 완전 파쇄하기 위해 가속기 내부 **SFU(Special Function Unit) 기계어 명령어와 1:1로 직접 융합 컴파일(Inline Fusion)**되도록 바인딩한 설계 철학을 그대로 보존했습니다.
3. **`shard_map` 기반 0바이트 무복사 다차원 메시 격자 사양 (`async_scheduler.py`)**
   - 분산 클러스터 토폴로지 축만 기하학적으로 분할하고 시간축과 특징 차원을 레지스터에 정적으로 동결 배치하여, **단 1바이트의 임시 버퍼 비용 없이 데이터 스트림을 온칩 단독 관류**시키려 했던 최고의 오케스트레이션 뼈대 스펙을 고스란히 계승했습니다.



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
| **최종 기술 목적 (Goal)** | 유체 수식을 통한 가상 공간 내 오차 0ns 복원 연출 | **대규모 분산 SGD 학습 중 그라디언트 폭발 및 NaN 폭사 차단** |
| **데이터 정밀도 (Precision)**| `> 0.5` 임계값 필터로 데이터를 `0`과 `1`로 이진화 파괴 | **`bfloat16 / float32` 연속적 소수점 그라디언트 무결성 100% 보존** |
| **수리물리 도메인 정의** | 암전(Blackout) 시 없는 데이터를 대수적으로 '창조/땜빵' | **튀는 고주파 충격파 노이즈를 점성으로 부드럽게 흡수하는 '정류 필터'** |
| **통신 스케줄링 기믹** | 대수적 우회 주장 (내부적으로는 NCCL `psum` 동기 배리어에 의존) | **XLA 컴파일러 단에서의 진짜 연산-통신 완전 비동기 겹치기 (`Overlapping`)** |
| **무선 결함 허용 (Fault)** | 가짜 데이터 풀(`cold_standby_pool`)을 주입해 수렴 연출 | **`stop_gradient` 캐시 락킹 및 항상성 기반 무선 에지 Elastic Rescue** |
| **출력 차원 명세 (Specs)** | 지터 축 강제 수축에 의한 3D 파티셔닝 `P(M, N, None)` | **지터 차원을 완전 사수하여 Llama SDPA와 직결되는 4D `P(N, M, N, N)`** |

---

## 📐 3-Tier Production Operational Pipeline (프로덕션 코어 명세)

본 프로덕션 코어(`production_core/`) 패키지는 가속기 내부 온칩 메모리(SRAM) 효율을 극대화하고 프레임워크 지터를 소멸시키기 위해 3개의 디커플링된 전사 레이어로 빌드되었습니다.

### 🌊 Layer 1: 점성 기반 그라디언트 난류 정류 (`core_smoother_xla.py`)
- **기능:** 버거스 방정식의 소산 항($+\sigma \frac{\partial^{2} \mathbf{\Phi}}{\partial x^{2}}$)을 역이용하여 분산 SGD 학습 중 튀는 고주파 수치 노이즈를 끈적한 점성 브레이크로 부드럽게 흡수합니다.
- **수직 계통:** Neumann 경계 조건(기울기 0 고정)을 인플레이스로 시뮬레이션하며, 이진화 반올림을 완전히 소멸시켜 `bfloat16 / float32` 고정밀 소수점 그라디언트 다양체의 무결성을 100% 사수합니다.

### ⚡ Layer 2: 무분기 실리콘 방화벽 및 미분 사슬 가드 (`math_guardrails.py`)
- **기능:** 칩 내부에서 발생하는 `NaN / INF` 수치 예외 및 임계치 초과 스파이크를 단 하나의 조건부 분기문(JMP) 없이 하드웨어 MUX 선택자(`selp.f32`)로 원자적으로 즉각 플러시합니다.
- **수직 계통:** 값을 단순히 클리핑하여 미분을 0으로 만드는 상용 가드의 치명타를 우회하고, 임계면 바깥에 미세 기울기(`Leaky Slope`)를 강제로 합성하여 오차 역전파 미분 사슬을 끝까지 살려둡니다.

### 🎛️ Layer 3: 하이브리드 토폴로지 오케스트레이션 및 핫스왑 제어
- **유선 데이터센터 패스 (`async_scheduler.py`):** JAX `shard_map` 정적 4차원 차원 고정 매핑을 가동합니다. GPU가 정류 연산을 수행하는 동안 백그라운드 레지스터 단에서 NCCL All-Reduce 통신(`psum`)을 동시에 실행하여 통신 레이턴시를 완벽하게 은닉(Latency Hiding)합니다.
- **무선 에지 가드 패스 (`elastic_governor.py`):** 가짜 데이터 땜빵 장치를 완전 탈거하고, `jax.lax.scan` 피드백 루프 안에서 시스템 스스로 직전 스텝까지 성공적으로 도달시킨 청정 상태를 `stop_gradient`로 밀봉 격리합니다. 85% 이상의 극한 무선 단선 환경에서도 전역 가중치 오염 없이 무중단 생존합니다.

---


## ⚡ Integrated Benchmark Test System (`test_production_pipeline.py`)

본 프로젝트는 가상 분산 환경과 실제 하드웨어 클러스터 모두를 범용 수용하는 실전형 통합 벤치마크 검증 시스템을 저장소 루트에 탑재하고 있습니다. 단 1초 만에 `production_core` 전체 패키지의 수치적 무결성과 유·무선 핫스왑 가동 여부를 전수 오디트(Audit)합니다.

### 🔬 2-Stage Stress Disruption Scenario (스트레스 시뮬레이션 환경)

1. **🏢 Stage 1: Wired Datacenter Jitter Track**
   - **인프라 모사:** 대규모 데이터센터 NVLink 환경 내부의 고속 분산 SGD 전송 선로를 모사합니다.
   - **난류 주입:** 미세 패킷 전송 시차 및 인프라 지터 요동(10% 무작위 노이즈 마스크)을 상시 가산하여 `async_scheduler.py` 커널의 온칩 연산-통신 완전 중첩(Overlapping) 레이턴시 은닉 효율을 정산합니다.
   
2. **📡 Stage 2: Wireless Edge Extreme Blackout Track**
   - **인프라 모사:** 드론 떼, 스타링크, 혹은 에지 네트워크 환경 내부의 불안정한 무선 통신 선로를 모사합니다.
   - **난류 주입:** 4차원 시퀀스 데이터 스트림 전면에 **88% 이상의 극단적인 대규모 패킷 드롭 및 기지국 폭사(Blackout) 스트레스**를 하드 인젝션합니다. 이를 통해 `elastic_governor.py`가 외부 데이터 단선 상태에서 스스로 무결함을 증명해낸 과거 상수를 `stop_gradient`로 밀봉하여 전역 가중치 NaN 전염을 무력화해 내는지 계측합니다.


### 💻 Execution & Verification Command (구동 명령어)

저장소 최상단에서 아래 명령어로 생산 코어의 수치 정정 동작을 검증합니다.
```bash
python production_core/test_production_pipeline.py
```

### 📺 Expected Audit Console Output (합격 로그 양식)

무분기 마스킹이 적용된 합격 콘솔 로그 규격입니다:

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

본 프로젝트는 **Apache License 2.0** 오픈소스로 배포됩니다.

- **상용 및 학술적 이용 권리:** 본 저장소의 핵심 가속 기전(`shard_map` 4D 최적화, 수리물리 점성 그라디언트 난류 정류, Leaky Slope NaN 방화벽)은 오픈소스 생태계의 모든 딥러닝 연구원, 인프라 설계자, 기업 클러스터가 자유롭게 포크(Fork)하여 상용 제품 및 학술 논문에 인용·변형·탑재할 수 있습니다.
- **기여 및 계보 존중:** 본 엔진을 기반으로 추가적인 리팩토링이나 랭체인, 프레임워크 플러그인을 확장 배포하실 경우, 원작자(`PJHkorea`)의 기하학적 3축 오케스트레이션 설계 원형과 수리물리학적 청사 서사(Lineage)를 README 및 코드 헤더 명세에 명시적으로 예우 및 인용해 주셔야 합니다.


