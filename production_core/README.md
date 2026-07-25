# 📡 FNG V3 Production Core: Developer Reference

This directory contains the **JAX/XLA-driven hybrid infrastructure kernel suite** architected for the radical synchronization and execution optimization of distributed LLM training and inference pipelines. By mapping continuous fluid dynamics formulations directly onto the accelerated control plane, this architecture guarantees extreme numerical stability and optimal communication-computation overlapping efficiency.

## 📂 Core Module Architecture
```directory
production_core/
├── __init__.py                     # Static namespace freezing & zero-latency runtime lookup gate
├── core_smoother_xla.py            # Gradient turbulence rectification via Burgers' viscous dissipation
├── math_guardrails.py              # Branchless NaN/INF silicon firewall with Leaky Slope autograd preservation
├── async_scheduler.py              # [🏢 Wired] shard_map-driven computation-communication (All-Reduce) overlap scheduler
├── elastic_governor.py             # [📡 Wireless] jax.lax.scan-bound resilient governor for fault-tolerant edge recovery
├── pytorch_mega_adapter.py         # Zero-copy DLPack memory tunneling bridge between PyTorch and JAX/XLA
├── transformer_fused.py            # Unified hybrid orchestration adapter interlocked with standard Llama 3D SDPA rails
├── test_production_pipeline.py     # Stress-fault injection launcher validating real-time numerical integrity
└── test_megatron_speedup_report.py # Megatron-spec TFLOPS & latency profiling engine (Llama-3-70B/DeepSeek-V3 aligned)
```


## 🛠️ Key API Interface (`transformer_fused.py`)
`FngInterleavedLlamaAttention` isolates the accelerated computing layer from host-side abstraction leaks. By eagerly pre-compiling both wired and wireless acceleration factories during initialization, this interface eliminates runtime XLA tracer re-trace bottlenecks.

```python
from production_core.transformer_fused import FngInterleavedLlamaAttention

# 🏗️ Bind distributed topology mesh and freeze optimizer graph during instantiation
fng_gate = FngInterleavedLlamaAttention(
    devices_mesh=devices_mesh, 
    mesh_axis_name="fluidic_mesh"
)

# ⚡ Zero-overhead runtime hot-swap via static compiler branch elimination
# The hardware path is determined at compile time based on the deploy_env layout flag.
context_vector = fng_gate(
    local_q=local_q,               # Query [Batch, Head_Dim, Feature_Dim] (Standard 3D Llama Spec)
    local_k=local_k,               # Key   [Batch, Jitter_Dim, Feature_Dim] (Wired V1: 3D / Wireless V2: 4D)
    local_v=local_v,               # Value [Batch, Jitter_Dim, Feature_Dim] (Wired V1: 3D / Wireless V2: 4D)
    deploy_env="WIRED_DATACENTER", # Target Engine Path: "WIRED_DATACENTER" or "WIRELESS_EDGE"
    current_drop_rate=0.0          # Real-time packet telemetry loss metrics (utilized for V2 Edge Care)
)
```


## 🏢 Distributed Sharding Layout (`async_scheduler.py`)
To preserve volatile continuous floating-point jitter dimensions with absolute fidelity, the core engine enforces a 4D `PartitionSpec` configured as `P(None, mesh_axis_name, None, None)`. This architecture guarantees zero-copy, static multi-axis sharding across the distributed accelerator mesh, bypassing runtime memory reallocation overheads entirely.

---

## ⛓️ Downstream Numerical Pipeline Chain (Vertical Data Flow)

The execution chain within this package is statically fused into a single unified XLA machine-code computation graph, permanently removing heap allocation latencies and host-to-device context-switching overheads.

1. **`core_smoother_xla.py` [Numerical Rectification Layer]**
   - Captures high-frequency fluctuations across the 4D gradient manifold via the viscous dissipation term of Burgers' equation, executing atomic algebraic skewness flattening through second- and third-order moment inversions.
2. **`math_guardrails.py` [Numerical Stabilization Layer]**
   - Traps arithmetic anomalies (`NaN / INF`) trickling from the rectification stage to instantly flush infected coordinates into a safe baseline (`0.0f`) via branchless hardware MUX selectors, while continuously synthesizing a `leaky_slope` gradient beyond the threshold manifold to prevent autograd chain deterioration.
3. **Hot-Swap Interlock Interface (`async_scheduler.py` / `elastic_governor.py`)**
   - **Wired Pass (`WIRED_DATACENTER`):** Co-schedules the execution layout of the fluidic smoother kernel and back-end NCCL All-Reduce collectives via `jax.lax.psum`, achieving near-perfect communication latency hiding inside the accelerator register lines.
   - **Wireless Pass (`WIRELESS_EDGE`):** Triggers an autograd isolation valve upon crossing the critical 85% packet dropout threshold, activating `jax.lax.stop_gradient` to isolate and freeze the historical clean state as a static checkpoint, ensuring invariant system homeostasis against total transmission blackouts.

---


## ⚠️ Accelerator Compiler Considerations

When移植 (porting) this core into external infrastructures or custom transformer layers, you must strictly enforce the following execution invariants to guarantee the architectural stability of the static XLA back-end compiler compilation passes:

1. **Static Device Mesh Topology Freezing (`devices_mesh`)**
   - Due to the eager, global tracing characteristics of the JAX/XLA compiler factory, any dynamic modification of the device mesh axis configurations or the overall accelerator topology layout during runtime will break the trace invariants. This triggers catastrophic compiler re-trace stalls. You must explicitly define, freeze, and pass a completely static `Mesh` instance during the adapter instantiation stage (`__init__`).
2. **4D Manifold Dimensional Vector Alignment (`PartitionSpec`)**
   - Both the `async_scheduler` and `elastic_governor` kernels uniformly share a 4D partitioning signature structured as `P(None, mesh_axis_name, None, None)` to guarantee the flawless atomic switching of runtime network pathways. Arbitrarily squeezing or unsqueezing dimensions via external KV-cache pipelines or higher-level frameworks will trigger terminal multi-axis alignment crashes (`ValueError: axes do not match`). You must strictly maintain a constant, invariant tensor rank across all I/O interfaces.

---


## 🛠️ PyTorch Framework Integration (`pytorch_mega_adapter.py`)
To enable seamless native integration with hyperscale distributed computing backends such as `Megatron-LM` and `DeepSeek-V3` context parallelism pipelines, this suite provides an optimized `torch.nn.Module` wrapper. `FngPyTorchMegaAdapter` utilizes the cross-framework DLPack shared memory interface to execute atomic, zero-copy pointer view mapping between PyTorch execution tracks and JAX/XLA hardware contexts.

```python
from production_core.pytorch_mega_adapter import FngPyTorchMegaAdapter

# 🏗️ Initialize the adapter module within the PyTorch distributed execution graph
fng_torch_adapter = FngPyTorchMegaAdapter(
    devices_mesh=devices_mesh, 
    mesh_axis_name="fluidic_mesh"
)

# ⚡ Execute zero-copy gradient streaming within the PyTorch forward attention block
# Utilizes an internal asynchronous barrier lock to align runtime execution tracks without host-side stalles.
output_context = fng_torch_adapter(
    pytorch_q=q_tensor,             # PyTorch Tensor [Batch, Seq_Len, Hidden_Dim] (Native Layout)
    pytorch_k=k_tensor,             # 3D Tensor for Wired V1 / 4D Sequence Tensor for Wireless V2
    pytorch_v=v_tensor,             # 3D Tensor for Wired V1 / 4D Sequence Tensor for Wireless V2
    pytorch_pollution_mask=mask,    # Volatile noise mask allocated inside PyTorch GPU memory space
    deploy_env="WIRED_DATACENTER"   # Execution routing: "WIRED_DATACENTER" or "WIRELESS_EDGE"
)
```


## 📊 Megatron Speedup Profiler (`test_megatron_speedup_report.py`)
A standalone, bare-metal accelerator profiling launcher designed to audit and calculate numerical rectification throughput and communication overlapping efficiency down to nanosecond precision under hyperscale parallel configurations.

- **Hardware-Level Silicon Clock Synchronization:** Eliminates Python host-side garbage collection (GC) latency artifacts and interpreter jitter by directly orchestrating low-level `torch.cuda.Event` hardware timers, extracting the invariant minimum hardware execution latency.
- **Quantitative TFLOPS and Hiding Rate Inversion:** Cross-references the captured silicon timings with standard Llama algebraic workload equations (FLOPs) to derive pure accelerator TFLOPS throughput, mapping the communication overlapping hiding efficiency ($\Delta$ %) against conventional synchronized NCCL barriers directly onto a high-density operational telemetry dashboard.

---

## 📜 Copyright & Copyleft
- **Distributed under the Apache License 2.0.**
- When forking, transcribing, or integrating the core mathematical fluidic filters and 4D `shard_map` orchestration topology blueprints of this framework into external architectures, you must explicitly preserve and credit the original engineering lineage of the author (**PJHkorea**) within all source code headers, documentation references, and derivative technical specifications.
