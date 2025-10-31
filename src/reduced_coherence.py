#!/usr/bin/env python3
"""
Reduced Source Coherence Module

Applies principles from partially coherent pulse propagation in nonlinear 
dispersive media to Monte Carlo integration and stochastic sampling for 
geometric factorization.

Inspired by optical physics findings where reduced source coherence 
unexpectedly improves robustness against temporal spreading, this module 
implements controlled "incoherence" mechanisms to enhance stability and 
success rates in high-dimensional geometric embeddings and lattice-based 
searches.

Key Concepts:
1. **Coherence Control**: Parameter α ∈ [0, 1] controls correlation in samples
   - α = 1.0: Fully coherent (standard QMC, maximum correlation)
   - α = 0.0: Fully incoherent (pure random, minimum correlation)
   - α ∈ (0, 1): Partially coherent (hybrid approaches)

2. **Ensemble Averaging**: Multiple independent sampling ensembles combined
   to simulate partial coherence (analogous to complex screen method)

3. **Split-Step Evolution**: Iterative refinement of candidates with 
   controlled decoherence at each step

4. **Variance Stabilization**: Lower coherence → reduced variance amplification
   in high-dimensional spaces (analogous to dispersion resistance)

Mathematical Framework:
- Coherence length: l_c ~ 1/α (correlation distance in sample space)
- Ensemble size: N_e ~ α^(-2) (number of independent realizations)
- Decoherence rate: γ ~ (1 - α) (per-step randomness injection)

References:
- arXiv:2503.02629: Partially coherent pulses in nonlinear dispersive media
- Physical Review A: Self-reconstruction robustness via reduced coherence
- Photonics: Complex screen method for numerical pulse propagation
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
from enum import Enum

try:
    from low_discrepancy import SobolSampler, GoldenAngleSampler
    LOW_DISCREPANCY_AVAILABLE = True
except ImportError:
    LOW_DISCREPANCY_AVAILABLE = False


class CoherenceMode(Enum):
    """Coherence control modes for sampling."""
    FULLY_COHERENT = "fully_coherent"      # α = 1.0 (standard QMC)
    PARTIALLY_COHERENT = "partially_coherent"  # α ∈ (0.3, 0.8)
    REDUCED_COHERENT = "reduced_coherent"   # α ∈ (0.1, 0.3)
    INCOHERENT = "incoherent"              # α ≈ 0.0 (pure random)
    ADAPTIVE = "adaptive"                   # α varies by iteration


@dataclass
class CoherenceMetrics:
    """Metrics for tracking coherence effects."""
    alpha: float  # Coherence parameter
    variance: float  # Sample variance
    correlation_length: float  # Effective correlation distance
    ensemble_diversity: float  # Diversity across ensembles
    success_rate: float  # Candidate hit rate
    

class ReducedCoherenceSampler:
    """
    Implements reduced coherence sampling for geometric factorization.
    
    Provides controlled decoherence mechanisms to enhance robustness
    against variance amplification in high-dimensional Monte Carlo
    integration and candidate search.
    """
    
    def __init__(self, 
                 seed: int = 42,
                 coherence_alpha: float = 0.5,
                 num_ensembles: int = 4):
        """
        Initialize reduced coherence sampler.
        
        Args:
            seed: Random seed for reproducibility
            coherence_alpha: Coherence parameter α ∈ [0, 1]
                           0.0 = fully incoherent (pure random)
                           1.0 = fully coherent (standard QMC)
            num_ensembles: Number of independent ensembles for averaging
        """
        if not 0.0 <= coherence_alpha <= 1.0:
            raise ValueError(f"coherence_alpha must be in [0, 1], got {coherence_alpha}")
        if num_ensembles < 1:
            raise ValueError(f"num_ensembles must be >= 1, got {num_ensembles}")
            
        self.seed = seed
        self.alpha = coherence_alpha
        self.num_ensembles = num_ensembles
        self.rng = np.random.Generator(np.random.PCG64(seed))
        
        # Derived parameters
        self.correlation_length = 1.0 / max(self.alpha, 0.01)
        self.decoherence_rate = 1.0 - self.alpha
        
    def sample_with_reduced_coherence(self,
                                     N: int,
                                     num_samples: int,
                                     base_sampler: str = "qmc") -> np.ndarray:
        """
        Generate samples with controlled coherence reduction.
        
        Args:
            N: Target number for factorization context
            num_samples: Number of samples to generate
            base_sampler: Base sampler type ("qmc", "sobol", "golden", "uniform")
            
        Returns:
            Array of samples with reduced coherence
        """
        sqrt_N = int(np.sqrt(N))
        samples = []
        
        # Generate samples from multiple ensembles
        samples_per_ensemble = num_samples // self.num_ensembles
        
        for ensemble_idx in range(self.num_ensembles):
            # Each ensemble gets independent seed
            ensemble_seed = self.seed + ensemble_idx * 1000
            
            if base_sampler == "qmc" or base_sampler == "sobol":
                # QMC with controlled randomization
                if LOW_DISCREPANCY_AVAILABLE:
                    sampler = SobolSampler(dimension=1, scramble=True, seed=ensemble_seed)
                    base_samples = sampler.generate(n=samples_per_ensemble)[:, 0]
                else:
                    # Fallback to uniform
                    rng = np.random.Generator(np.random.PCG64(ensemble_seed))
                    base_samples = rng.uniform(0, 1, samples_per_ensemble)
            elif base_sampler == "golden":
                # Golden-angle with scrambling
                if LOW_DISCREPANCY_AVAILABLE:
                    sampler = GoldenAngleSampler(seed=ensemble_seed)
                    base_samples = sampler.generate_1d(n=samples_per_ensemble, offset=ensemble_idx)
                else:
                    rng = np.random.Generator(np.random.PCG64(ensemble_seed))
                    base_samples = rng.uniform(0, 1, samples_per_ensemble)
            else:  # uniform
                rng = np.random.Generator(np.random.PCG64(ensemble_seed))
                base_samples = rng.uniform(0, 1, samples_per_ensemble)
            
            # Apply coherence reduction (mix with random noise)
            noise = self.rng.uniform(0, 1, samples_per_ensemble)
            coherence_samples = self.alpha * base_samples + (1 - self.alpha) * noise
            
            # Map to candidate range around sqrt(N)
            spread = sqrt_N * 0.05
            candidates = sqrt_N + (coherence_samples - 0.5) * 2 * spread
            
            samples.extend(candidates)
        
        return np.array(samples)
    
    def ensemble_averaged_sampling(self,
                                   N: int,
                                   num_samples: int,
                                   phi_bias: bool = True) -> List[int]:
        """
        Ensemble averaging for candidate generation (complex screen analog).
        
        Generates multiple independent sample sets and combines them
        to simulate partial coherence effects. Each ensemble represents
        a different realization of the sampling process.
        
        Args:
            N: Number to factor
            num_samples: Total samples across all ensembles
            phi_bias: Apply golden ratio bias
            
        Returns:
            Combined candidate list with ensemble averaging
        """
        sqrt_N = int(np.sqrt(N))
        all_candidates = []
        
        samples_per_ensemble = num_samples // self.num_ensembles
        phi = (1 + np.sqrt(5)) / 2  # Golden ratio
        
        for ensemble_idx in range(self.num_ensembles):
            ensemble_seed = self.seed + ensemble_idx * 1000
            rng = np.random.Generator(np.random.PCG64(ensemble_seed))
            
            # Generate base samples
            base_samples = rng.uniform(-1, 1, samples_per_ensemble)
            
            # Apply decoherence (controlled randomization)
            if self.alpha < 1.0:
                noise = rng.normal(0, self.decoherence_rate, samples_per_ensemble)
                base_samples = base_samples + noise
                base_samples = np.clip(base_samples, -1, 1)
            
            # Convert to candidates
            for i, sample in enumerate(base_samples):
                if phi_bias:
                    # Apply φ modulation with reduced coherence
                    phi_mod = (i % phi) / phi
                    offset_scale = phi_mod ** 0.3
                    offset_scale *= (1 + self.decoherence_rate * rng.normal(0, 0.1))
                    offset = int(sqrt_N * 0.05 * offset_scale * np.sign(sample))
                else:
                    offset = int(sqrt_N * 0.05 * sample)
                
                candidate = sqrt_N + offset
                
                if candidate > 1 and candidate < N:
                    all_candidates.append(candidate)
        
        # Remove duplicates while preserving diversity
        return sorted(set(all_candidates))
    
    def split_step_evolution(self,
                            N: int,
                            initial_candidates: List[int],
                            num_steps: int = 5,
                            refinement_factor: float = 0.8) -> List[int]:
        """
        Split-step evolution with controlled decoherence.
        
        Iteratively refines candidates with decoherence injection at each
        step, analogous to split-step Fourier propagation in nonlinear optics.
        
        Args:
            N: Number to factor
            initial_candidates: Starting candidate set
            num_steps: Number of refinement steps
            refinement_factor: Spread reduction per step
            
        Returns:
            Refined candidate list
        """
        candidates = set(initial_candidates)
        sqrt_N = int(np.sqrt(N))
        
        for step in range(num_steps):
            # Adaptive decoherence: increase randomness over iterations
            step_alpha = self.alpha * (refinement_factor ** step)
            step_decoherence = 1.0 - step_alpha
            
            new_candidates = set()
            
            for candidate in candidates:
                # Local refinement with decoherence
                offset_range = int(sqrt_N * 0.01 * (1 + step))
                
                # Add coherent neighbors
                for offset in range(-offset_range, offset_range + 1):
                    new_cand = candidate + offset
                    if new_cand > 1 and new_cand < N:
                        new_candidates.add(new_cand)
                
                # Inject decoherence (random jumps)
                if step_decoherence > 0.1:
                    num_jumps = max(1, int(5 * step_decoherence))
                    for _ in range(num_jumps):
                        random_offset = int(self.rng.normal(0, sqrt_N * 0.02))
                        jump_cand = candidate + random_offset
                        if jump_cand > 1 and jump_cand < N:
                            new_candidates.add(jump_cand)
            
            candidates = new_candidates
            
            # Limit size to prevent explosion
            if len(candidates) > 10000:
                candidates = set(sorted(candidates)[:10000])
        
        return sorted(candidates)
    
    def adaptive_coherence_sampling(self,
                                   N: int,
                                   num_samples: int,
                                   target_variance: float = 0.1) -> Tuple[List[int], List[float]]:
        """
        Adaptive coherence control based on observed variance.
        
        Dynamically adjusts coherence parameter to maintain target
        variance, preventing runaway amplification in high dimensions.
        
        Args:
            N: Number to factor
            num_samples: Number of samples
            target_variance: Target variance threshold
            
        Returns:
            (candidates, alpha_history)
        """
        sqrt_N = int(np.sqrt(N))
        candidates = []
        alpha_history = []
        
        current_alpha = self.alpha
        batch_size = num_samples // 10  # Process in batches
        
        for batch_idx in range(10):
            # Generate batch with current coherence
            temp_sampler = ReducedCoherenceSampler(
                seed=self.seed + batch_idx,
                coherence_alpha=current_alpha,
                num_ensembles=self.num_ensembles
            )
            
            batch_samples = temp_sampler.sample_with_reduced_coherence(
                N, batch_size, base_sampler="qmc"
            )
            
            # Convert to candidates
            batch_candidates = [int(s) for s in batch_samples if 1 < s < N]
            candidates.extend(batch_candidates)
            
            # Measure variance
            if len(batch_candidates) > 1:
                candidate_variance = np.var(batch_candidates) / (sqrt_N ** 2)
                
                # Adapt coherence to maintain target variance
                if candidate_variance > target_variance:
                    # Too much variance → reduce coherence (more randomness)
                    current_alpha *= 0.9
                elif candidate_variance < target_variance * 0.5:
                    # Too little variance → increase coherence (more structure)
                    current_alpha = min(1.0, current_alpha * 1.1)
                
                current_alpha = np.clip(current_alpha, 0.1, 1.0)
            
            alpha_history.append(current_alpha)
        
        return sorted(set(candidates)), alpha_history
    
    def compute_metrics(self,
                       candidates: List[int],
                       N: int,
                       true_factors: Optional[Tuple[int, int]] = None) -> CoherenceMetrics:
        """
        Compute coherence metrics for analysis.
        
        Args:
            candidates: Generated candidate list
            N: Target number
            true_factors: Optional known factors for success rate
            
        Returns:
            CoherenceMetrics object
        """
        sqrt_N = int(np.sqrt(N))
        
        # Variance (normalized)
        variance = np.var(candidates) / (sqrt_N ** 2) if candidates else 0.0
        
        # Correlation length (from autocorrelation)
        if len(candidates) > 10:
            sorted_cands = sorted(candidates)
            diffs = np.diff(sorted_cands)
            correlation_length = np.mean(diffs) / sqrt_N if len(diffs) > 0 else 0.0
        else:
            correlation_length = 0.0
        
        # Ensemble diversity (coefficient of variation)
        ensemble_diversity = np.std(candidates) / np.mean(candidates) if candidates and np.mean(candidates) > 0 else 0.0
        
        # Success rate
        success_rate = 0.0
        if true_factors is not None:
            p, q = true_factors
            if p in candidates or q in candidates:
                success_rate = 1.0
        
        return CoherenceMetrics(
            alpha=self.alpha,
            variance=variance,
            correlation_length=correlation_length,
            ensemble_diversity=ensemble_diversity,
            success_rate=success_rate
        )


def compare_coherence_modes(N: int,
                           num_samples: int = 1000,
                           true_factors: Optional[Tuple[int, int]] = None,
                           seed: int = 42) -> Dict[str, Any]:
    """
    Compare different coherence modes for factorization.
    
    Args:
        N: Number to factor
        num_samples: Samples per mode
        true_factors: Known factors for validation
        seed: Random seed
        
    Returns:
        Dictionary with results for each mode
    """
    results = {}
    
    # Test different coherence levels
    coherence_levels = {
        "fully_coherent": 1.0,
        "high_coherent": 0.8,
        "moderate_coherent": 0.5,
        "reduced_coherent": 0.2,
        "incoherent": 0.05
    }
    
    for mode_name, alpha in coherence_levels.items():
        sampler = ReducedCoherenceSampler(
            seed=seed,
            coherence_alpha=alpha,
            num_ensembles=4
        )
        
        # Generate candidates
        candidates = sampler.ensemble_averaged_sampling(N, num_samples, phi_bias=True)
        
        # Compute metrics
        metrics = sampler.compute_metrics(candidates, N, true_factors)
        
        results[mode_name] = {
            "alpha": alpha,
            "num_candidates": len(candidates),
            "unique_candidates": len(set(candidates)),
            "metrics": metrics,
            "candidates": candidates[:50]  # Store sample for inspection
        }
    
    return results


if __name__ == "__main__":
    print("=" * 70)
    print("Reduced Source Coherence Demonstration")
    print("=" * 70)
    
    # Test case: N = 899 = 29 × 31
    N = 899
    true_factors = (29, 31)
    
    print(f"\nTarget: N = {N} = {true_factors[0]} × {true_factors[1]}")
    print(f"sqrt(N) = {int(np.sqrt(N))}")
    
    # Compare coherence modes
    print("\n" + "-" * 70)
    print("Comparing Coherence Modes")
    print("-" * 70)
    
    results = compare_coherence_modes(N, num_samples=500, true_factors=true_factors)
    
    print(f"\n{'Mode':<20} {'Alpha':<8} {'Candidates':<12} {'Variance':<12} {'Success'}")
    print("-" * 70)
    
    for mode_name, result in results.items():
        metrics = result["metrics"]
        print(f"{mode_name:<20} {result['alpha']:<8.2f} {result['num_candidates']:<12} "
              f"{metrics.variance:<12.6f} {'✓' if metrics.success_rate > 0 else '✗'}")
    
    # Demonstrate split-step evolution
    print("\n" + "-" * 70)
    print("Split-Step Evolution with Decoherence")
    print("-" * 70)
    
    sampler = ReducedCoherenceSampler(seed=42, coherence_alpha=0.5, num_ensembles=4)
    initial = sampler.ensemble_averaged_sampling(N, 100, phi_bias=True)
    print(f"Initial candidates: {len(initial)}")
    
    evolved = sampler.split_step_evolution(N, initial, num_steps=3)
    print(f"After evolution: {len(evolved)}")
    
    if true_factors[0] in evolved or true_factors[1] in evolved:
        print("✓ Factors found in evolved set!")
    
    # Demonstrate adaptive coherence
    print("\n" + "-" * 70)
    print("Adaptive Coherence Control")
    print("-" * 70)
    
    adaptive_sampler = ReducedCoherenceSampler(seed=42, coherence_alpha=0.8, num_ensembles=4)
    adaptive_cands, alpha_hist = adaptive_sampler.adaptive_coherence_sampling(N, 500)
    
    print(f"Total candidates: {len(adaptive_cands)}")
    print(f"Alpha evolution: {' → '.join([f'{a:.2f}' for a in alpha_hist[::2]])}")
    
    if true_factors[0] in adaptive_cands or true_factors[1] in adaptive_cands:
        print("✓ Factors found with adaptive coherence!")
    
    print("\n" + "=" * 70)
    print("Demonstration Complete")
    print("=" * 70)
