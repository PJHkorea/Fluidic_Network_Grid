import jax
import jax.numpy as jnp
from functools import partial

from production_core.core_smoother_xla import execute_gradient_viscous_smoother
from production_core.math_guardrails import enforce_algebraic_safety_gate

@partial(jax.jit, static_argnums=(2, 3))
def enforce_algebraic_safety_gate(purified_gradient, global_threshold=1e6, leaky_slope=1e-3, clean_baseline_val=0.0):
    """
    [FNG V3 PRODUCTION CORE - ALGEBRAIC SAFETY GATE & NAN FIREWALL]
    
    A production-grade numerical stabilization kernel that embeds the rigorous firewall mechanisms 
    introduced in `egregore-core-jax` directly into JAX/XLA native machine-code passes.
    This gate atomically flushes terminal arithmetic anomalies (`NaN/INF`) and extreme spikes 
    without introducing a single conditional branch instruction (`if-else`/`JMP`), while concurrently 
    synthesizing a fractional attenuation slope (`Leaky Slope`) beyond the threshold boundary to prevent 
    the catastrophic disruption and freezing of the autograd gradient chain.
    
    Args:
        purified_gradient: Fully rectified continuous floating-point gradient tensor originating 
                           from the `core_smoother_xla` computational pipeline.
        global_threshold: The absolute threshold scale used to isolate and classify extreme 
                          numerical explosion boundaries (Default: 1.0e6).
        leaky_slope: The fine-grained attenuation gradient injected beyond the threshold manifold 
                     to conserve and rescue the backpropagation autograd chain (Default: 0.001).
        clean_baseline_val: The low-voltage logical baseline sign used to securely reset 
                            complete arithmetic breakdowns and fatal invalid values (Default: 0.0).
        
    Returns:
        stabilized_gradient: Clean production gradient tensor with absolute numerical stability verified 
                             and the backward derivative pathway fully preserved.
    """
    target_dtype = purified_gradient.dtype
    threshold_tensor = jnp.array(global_threshold, dtype=target_dtype)
    slope_tensor = jnp.array(leaky_slope, dtype=target_dtype)
    baseline_tensor = jnp.array(clean_baseline_val, dtype=target_dtype)


       # --------------------------------------------------------------------------
    # ⚡ STEP 1: Hardware-Native NaN / INF Artifact Capture (Bitwise Exception Isolation)
    # --------------------------------------------------------------------------
    # Combines `jnp.isnan` and `jnp.isinf` to trap lower-level silicon arithmetic anomalies.
    # Completely avoids conditional branches, isolating infected coordinate spaces into a 1.0 (True) bitmask matrix.
    invalid_mask = jnp.isnan(purified_gradient) | jnp.isinf(purified_gradient)
    invalid_mask_float = invalid_mask.astype(target_dtype)

    # Primary Atomic Purification: Instantly flushes fatal invalid coordinates into the safe baseline logic rail (0.0f).
    # Achieves 0ns runtime flattening by mapping the decision path directly onto hardware MUX selection primitives (`jax.lax.select`).
    purged_step_1 = jax.lax.select(invalid_mask, jnp.full_like(purified_gradient, baseline_tensor), purified_gradient)

    # --------------------------------------------------------------------------
    # 🎛️ STEP 2: Leaky Slope Mapping for Extreme Spikes (Autograd Chain Preservation)
    # --------------------------------------------------------------------------
    # Tracks and isolates extreme scalar spikes that exceed the absolute global explosion threshold boundary (1e6).
    abs_gradient = jnp.abs(purged_step_1)
    overflow_mask = abs_gradient > threshold_tensor

    # [★CRITICAL EGREGORE PIVOT★] 
    # Hard-clipping tensors (`jnp.clip`) into constants forces their local derivatives to zero, permanently freezing backpropagation.
    # To bypass this limitation, this engine derives the excess delta beyond the threshold boundary and synthesizes 
    # an active `leaky_slope` multiplier, keeping the backpropagation autograd chain fully functional and responsive.
    sign_mask = jnp.sign(purged_step_1)
    excess_delta = abs_gradient - threshold_tensor
    
    # Intramanifold Boundary = Preserves raw values / Extramanifold Boundary = Threshold + (Excess * leaky_slope) with sign conservation
    leaky_clamped_value = sign_mask * (threshold_tensor + (excess_delta * slope_tensor))
    
    # Finalizes the branchless execution track by resolving the overflow boundary via a single hardware-level MUX pass
    stabilized_gradient = jax.lax.select(overflow_mask, leaky_clamped_value, purged_step_1)

    return stabilized_gradient
