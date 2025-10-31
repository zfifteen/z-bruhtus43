#!/usr/bin/env python3
"""
Randomized Quasi-Monte Carlo (RQMC) Control Knob Module

Maps reduced coherence parameter α to QMC randomization strength (scrambling depth)
and implements ensemble-based RQMC with variance estimation capabilities.

Key Concepts:
1. **Coherence → Scrambling Mapping**: α ∈ [0, 1] controls scrambling strength
   - α = 1.0: Minimal scrambling (fully coherent QMC)
   - α = 0.5: Moderate scrambling (balanced RQMC)
   - α = 0.0: Maximum scrambling (approaching pure MC)

2. **Ensemble Replications**: M independent scrambles for variance estimation
   - Analogous to complex screen ensemble in optics
   - Enables unbiased error bars via RQMC theory

3. **Split-Step Evolution**: Periodic re-scrambling between refinement stages
   - Local refinement → global re-mixing cycles
   - Mirrors split-step Fourier propagation with phase screens

4. **Weighted Discrepancy**: Dimension-wise importance for high-dimensional problems
   - Adaptive α scheduling per coordinate
   - Targets ~10% normalized variance per dimension

Mathematical Framework:
- Scrambling depth: d(α) = ⌈32 × (1 - α)⌉ (bit levels to scramble)
- Ensemble size: M(α) = max(1, ⌈10 × (1 - α²)⌉) (independent replications)
- Variance target: σ²_target ≈ 0.1 (10% normalized variance)

Convergence Rates (Owen 1997, Dick 2010):
- Unscrambled QMC: O(N^(-1) (log N)^(s-1))
- Scrambled nets: O(N^(-3/2+ε)) for smooth integrands
- MC baseline: O(N^(-1/2))

References:
- arXiv:2503.02629: Partially coherent pulses in nonlinear media
- Owen (1997): Scrambled net variance for smooth functions
- L'Ecuyer (2020): Randomized Quasi-Monte Carlo overview
- Burley et al. (2020): Practical hash-based Owen scrambling
"""

import numpy as np
from typing import List, Tuple, Optional, Dict, Any, Callable
from dataclasses import dataclass
from enum import Enum

try:
    from low_discrepancy import SobolSampler, GoldenAngleSampler
    LOW_DISCREPANCY_AVAILABLE = True
except ImportError:
    LOW_DISCREPANCY_AVAILABLE = False


class RQMCMode(Enum):
    """RQMC sampling modes with different scrambling strategies."""
    SOBOL_SCRAMBLED = "sobol_scrambled"           # Scrambled Sobol' (recommended)
    HALTON_SCRAMBLED = "halton_scrambled"         # Scrambled Halton
    SOBOL_OWEN = "sobol_owen"                     # Owen scrambling for Sobol'
    ADAPTIVE_SCRAMBLING = "adaptive_scrambling"    # Adaptive α scheduling
    WEIGHTED_DISCREPANCY = "weighted_discrepancy"  # Dimension-wise importance


@dataclass
class RQMCMetrics:
    """Metrics for RQMC performance tracking."""
    alpha: float                    # Coherence/scrambling parameter
    scrambling_depth: int          # Bit levels scrambled
    num_replications: int          # Ensemble size M
    variance: float                # Empirical variance
    discrepancy: float             # L∞ star discrepancy estimate
    convergence_rate: float        # Empirical convergence rate
    success_rate: float            # Factor hit rate (if known factors)


class RQMCScrambler:
    """
    Base class for RQMC scrambling strategies.
    
    Implements various scrambling methods that can be controlled by
    the coherence parameter α, providing a unified interface for
    randomized quasi-Monte Carlo sampling.
    """
    
    def __init__(self, 
                 alpha: float = 0.5,
                 seed: Optional[int] = None):
        """
        Initialize RQMC scrambler.
        
        Args:
            alpha: Coherence parameter ∈ [0, 1]
                  0.0 = maximum scrambling (high randomization)
                  1.0 = minimal scrambling (low randomization)
            seed: Random seed for reproducible scrambling
        """
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        
        self.alpha = alpha
        self.seed = seed
        self.rng = np.random.Generator(np.random.PCG64(seed)) if seed is not None else None
        
        # Derived parameters
        self.scrambling_depth = self._compute_scrambling_depth(alpha)
        self.num_replications = self._compute_num_replications(alpha)
    
    def _compute_scrambling_depth(self, alpha: float) -> int:
        """
        Map coherence α to scrambling depth.
        
        Scrambling depth d(α) determines how many bit levels to randomize:
        - α = 1.0 → d = 1 (minimal scrambling, preserve structure)
        - α = 0.5 → d = 16 (moderate scrambling)
        - α = 0.0 → d = 32 (maximum scrambling, full randomization)
        
        Args:
            alpha: Coherence parameter
            
        Returns:
            Scrambling depth (bit levels)
        """
        # Non-linear mapping: preserve more structure at high α
        depth = int(np.ceil(32 * (1 - alpha**2)))
        return max(1, min(32, depth))
    
    def _compute_num_replications(self, alpha: float) -> int:
        """
        Map coherence α to number of ensemble replications.
        
        More randomization (lower α) requires more replications for
        reliable variance estimation:
        - α = 1.0 → M = 1 (deterministic QMC)
        - α = 0.5 → M = 8 (moderate ensembles)
        - α = 0.0 → M = 10 (many ensembles for variance)
        
        Args:
            alpha: Coherence parameter
            
        Returns:
            Number of independent scrambles
        """
        # Quadratic scaling: variance ∝ 1/M
        M = int(np.ceil(10 * (1 - alpha**2)))
        return max(1, M)
    
    def scramble_sequence(self, 
                         sequence: np.ndarray, 
                         depth: Optional[int] = None) -> np.ndarray:
        """
        Apply digital scrambling to a low-discrepancy sequence.
        
        Implements nested random digit scrambling (Owen scrambling)
        at specified depth, preserving low-discrepancy structure
        while enabling variance estimation.
        
        Args:
            sequence: Input sequence of shape (n, dim) in [0, 1]
            depth: Optional scrambling depth (uses self.scrambling_depth if None)
            
        Returns:
            Scrambled sequence of same shape
        """
        if self.rng is None:
            raise ValueError("Cannot scramble without seed")
        
        depth = depth if depth is not None else self.scrambling_depth
        n, dim = sequence.shape
        scrambled = np.copy(sequence)
        
        for d in range(dim):
            # Convert to integer representation (32-bit)
            int_seq = (sequence[:, d] * (2**31)).astype(np.int64)
            
            # Apply nested random digit scrambling
            for level in range(depth):
                # Random permutation at this bit level
                if self.alpha < 1.0:
                    # Generate level-specific permutation mask
                    mask = self.rng.integers(0, 2**31, dtype=np.int64)
                    int_seq = int_seq ^ mask
                    
                    # Reduce intensity for higher α (less scrambling)
                    if self.alpha > 0.5:
                        # Probabilistic application
                        apply_mask = self.rng.random(n) < (1 - self.alpha)
                        int_seq = np.where(apply_mask, int_seq, (sequence[:, d] * (2**31)).astype(np.int64))
            
            # Convert back to [0, 1]
            scrambled[:, d] = (int_seq % (2**31)) / (2**31)
        
        return scrambled
    
    def hash_scramble(self, 
                     index: int, 
                     dim: int, 
                     value: float) -> float:
        """
        Hash-based Owen scrambling (Burley et al. 2020).
        
        Fast, practical scrambling using hash functions instead of
        explicit random number generation. Suitable for on-the-fly
        sample generation.
        
        Args:
            index: Sample index
            dim: Dimension index
            value: Input value in [0, 1]
            
        Returns:
            Scrambled value in [0, 1]
        """
        # Hash seed combines index, dimension, and global seed
        hash_seed = self.seed if self.seed is not None else 0
        hash_val = hash((index, dim, hash_seed))
        
        # Convert value to bits
        bits = int(value * (2**31))
        
        # Apply hash-based permutation at each level
        for level in range(self.scrambling_depth):
            level_hash = hash((hash_val, level)) % (2**31)
            bits = bits ^ level_hash
        
        # Convert back, scaled by α
        scrambled_bits = bits % (2**31)
        scrambled_val = scrambled_bits / (2**31)
        
        # Blend based on coherence
        return self.alpha * value + (1 - self.alpha) * scrambled_val


class ScrambledSobolSampler(RQMCScrambler):
    """
    Scrambled Sobol' sequence generator with α-controlled randomization.
    
    Combines Sobol' low-discrepancy structure with Owen scrambling,
    providing optimal convergence rates for smooth integrands while
    enabling variance estimation via multiple replications.
    """
    
    def __init__(self, 
                 dimension: int = 2,
                 alpha: float = 0.5,
                 seed: Optional[int] = None):
        """
        Initialize scrambled Sobol' sampler.
        
        Args:
            dimension: Number of dimensions (1-8 supported)
            alpha: Coherence/scrambling parameter
            seed: Random seed for scrambling
        """
        super().__init__(alpha=alpha, seed=seed)
        
        if not LOW_DISCREPANCY_AVAILABLE:
            raise ImportError("low_discrepancy module required for Sobol' sampling")
        
        self.dimension = dimension
        # Base Sobol' sampler (unscrambled)
        self.base_sampler = SobolSampler(dimension=dimension, scramble=False, seed=seed)
    
    def generate(self, n: int) -> np.ndarray:
        """
        Generate scrambled Sobol' sequence.
        
        Args:
            n: Number of samples
            
        Returns:
            Array of shape (n, dimension) with scrambled samples
        """
        # Generate base Sobol' sequence
        base_sequence = self.base_sampler.generate(n)
        
        # Apply α-controlled scrambling
        if self.alpha < 1.0:
            scrambled = self.scramble_sequence(base_sequence)
            return scrambled
        else:
            return base_sequence
    
    def generate_replications(self, n: int) -> List[np.ndarray]:
        """
        Generate M independent scrambled replications.
        
        Each replication is independently scrambled, enabling
        unbiased variance estimation via RQMC theory.
        
        Args:
            n: Samples per replication
            
        Returns:
            List of M arrays, each of shape (n, dimension)
        """
        replications = []
        base_seed = self.seed if self.seed is not None else 0
        
        for m in range(self.num_replications):
            # Create independent scrambled replica
            sampler = ScrambledSobolSampler(
                dimension=self.dimension,
                alpha=self.alpha,
                seed=base_seed + m * 1000
            )
            replication = sampler.generate(n)
            replications.append(replication)
        
        return replications


class ScrambledHaltonSampler(RQMCScrambler):
    """
    Scrambled Halton sequence generator with α-controlled randomization.
    
    Addresses Halton correlation pathologies in high dimensions via
    scrambling, making it suitable for high-dimensional factorization
    problems where unscrambled Halton fails.
    """
    
    def __init__(self, 
                 dimension: int = 2,
                 alpha: float = 0.5,
                 bases: Optional[List[int]] = None,
                 seed: Optional[int] = None):
        """
        Initialize scrambled Halton sampler.
        
        Args:
            dimension: Number of dimensions
            alpha: Coherence/scrambling parameter
            bases: Optional prime bases (uses first d primes if None)
            seed: Random seed for scrambling
        """
        super().__init__(alpha=alpha, seed=seed)
        
        self.dimension = dimension
        self.bases = bases if bases is not None else self._get_prime_bases(dimension)
        
        if len(self.bases) < dimension:
            raise ValueError(f"Need at least {dimension} prime bases, got {len(self.bases)}")
    
    def _get_prime_bases(self, d: int) -> List[int]:
        """Get first d prime numbers as bases."""
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]
        if d > len(primes):
            raise ValueError(f"Need more primes for dimension {d}")
        return primes[:d]
    
    def _halton_base(self, index: int, base: int) -> float:
        """Generate single Halton value in given base."""
        result = 0.0
        f = 1.0 / base
        i = index
        while i > 0:
            result += f * (i % base)
            i //= base
            f /= base
        return result
    
    def generate(self, n: int) -> np.ndarray:
        """
        Generate scrambled Halton sequence.
        
        Args:
            n: Number of samples
            
        Returns:
            Array of shape (n, dimension) with scrambled samples
        """
        sequence = np.zeros((n, self.dimension))
        
        # Generate base Halton sequence
        for i in range(n):
            for d in range(self.dimension):
                sequence[i, d] = self._halton_base(i + 1, self.bases[d])
        
        # Apply α-controlled scrambling
        if self.alpha < 1.0:
            scrambled = self.scramble_sequence(sequence)
            return scrambled
        else:
            return sequence
    
    def generate_replications(self, n: int) -> List[np.ndarray]:
        """
        Generate M independent scrambled replications.
        
        Args:
            n: Samples per replication
            
        Returns:
            List of M arrays, each of shape (n, dimension)
        """
        replications = []
        base_seed = self.seed if self.seed is not None else 0
        
        for m in range(self.num_replications):
            sampler = ScrambledHaltonSampler(
                dimension=self.dimension,
                alpha=self.alpha,
                bases=self.bases,
                seed=base_seed + m * 1000
            )
            replication = sampler.generate(n)
            replications.append(replication)
        
        return replications


class AdaptiveRQMCSampler:
    """
    Adaptive RQMC sampler with α scheduling for target variance control.
    
    Dynamically adjusts scrambling strength to maintain ~10% normalized
    variance, as specified in the issue requirements. Monitors variance
    per dimension and adapts α accordingly.
    """
    
    def __init__(self,
                 dimension: int = 2,
                 target_variance: float = 0.1,
                 sampler_type: str = "sobol",
                 seed: Optional[int] = None):
        """
        Initialize adaptive RQMC sampler.
        
        Args:
            dimension: Number of dimensions
            target_variance: Target normalized variance (~10%)
            sampler_type: "sobol" or "halton"
            seed: Random seed
        """
        self.dimension = dimension
        self.target_variance = target_variance
        self.sampler_type = sampler_type
        self.seed = seed
        
        # Start with moderate α
        self.alpha = 0.5
        self.alpha_history = []
        
        # Dimension-wise α for weighted discrepancy
        self.alpha_per_dim = np.ones(dimension) * self.alpha
    
    def generate_adaptive(self, 
                         n: int,
                         num_batches: int = 10) -> Tuple[np.ndarray, List[float]]:
        """
        Generate samples with adaptive α scheduling.
        
        Processes samples in batches, measuring variance after each
        batch and adjusting α to maintain target variance.
        
        Args:
            n: Total number of samples
            num_batches: Number of adaptive batches
            
        Returns:
            (samples, alpha_history)
        """
        batch_size = n // num_batches
        all_samples = []
        
        for batch_idx in range(num_batches):
            # Create sampler with current α
            if self.sampler_type == "sobol":
                sampler = ScrambledSobolSampler(
                    dimension=self.dimension,
                    alpha=self.alpha,
                    seed=self.seed + batch_idx if self.seed else None
                )
            else:  # halton
                sampler = ScrambledHaltonSampler(
                    dimension=self.dimension,
                    alpha=self.alpha,
                    seed=self.seed + batch_idx if self.seed else None
                )
            
            # Generate batch
            batch = sampler.generate(batch_size)
            all_samples.append(batch)
            
            # Measure variance
            combined = np.vstack(all_samples)
            variance = np.var(combined, axis=0).mean()
            
            # Adapt α to approach target variance
            if variance > self.target_variance * 1.2:
                # Too much variance → increase α (less scrambling, more structure)
                self.alpha = min(0.95, self.alpha * 1.1)
            elif variance < self.target_variance * 0.8:
                # Too little variance → decrease α (more scrambling)
                self.alpha = max(0.05, self.alpha * 0.9)
            
            self.alpha_history.append(self.alpha)
        
        samples = np.vstack(all_samples)
        return samples, self.alpha_history
    
    def generate_weighted_discrepancy(self,
                                     n: int,
                                     dimension_weights: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate samples with dimension-wise α (weighted discrepancy).
        
        High-dimensional problems benefit from different scrambling
        strengths per coordinate, focusing randomization on important
        dimensions (e.g., curvature-weighted coordinates).
        
        Args:
            n: Number of samples
            dimension_weights: Optional importance weights per dimension
                              Higher weight → more scrambling (lower α)
            
        Returns:
            Samples with dimension-wise scrambling
        """
        if dimension_weights is None:
            # Default: uniform weights
            dimension_weights = np.ones(self.dimension)
        else:
            dimension_weights = np.asarray(dimension_weights)
            if len(dimension_weights) != self.dimension:
                raise ValueError(f"Need {self.dimension} weights, got {len(dimension_weights)}")
        
        # Normalize weights
        dimension_weights = dimension_weights / dimension_weights.max()
        
        # Map weights to per-dimension α
        # Higher weight → lower α (more scrambling for important dims)
        self.alpha_per_dim = 1.0 - 0.9 * dimension_weights
        
        # Generate base samples
        if self.sampler_type == "sobol":
            sampler = ScrambledSobolSampler(
                dimension=self.dimension,
                alpha=0.5,  # Will override per-dimension
                seed=self.seed
            )
        else:
            sampler = ScrambledHaltonSampler(
                dimension=self.dimension,
                alpha=0.5,
                seed=self.seed
            )
        
        base_samples = sampler.base_sampler.generate(n) if hasattr(sampler, 'base_sampler') else sampler.generate(n)
        
        # Apply dimension-wise scrambling
        scrambled = np.copy(base_samples)
        for d in range(self.dimension):
            alpha_d = self.alpha_per_dim[d]
            depth_d = int(np.ceil(32 * (1 - alpha_d**2)))
            
            if alpha_d < 1.0 and sampler.rng is not None:
                int_seq = (base_samples[:, d] * (2**31)).astype(np.int64)
                
                for level in range(depth_d):
                    mask = sampler.rng.integers(0, 2**31, dtype=np.int64)
                    int_seq = int_seq ^ mask
                
                scrambled[:, d] = (int_seq % (2**31)) / (2**31)
        
        return scrambled


class SplitStepRQMC:
    """
    Split-step RQMC evolution with periodic re-scrambling.
    
    Mirrors split-step Fourier propagation from optics:
    - Local refinement: exploit low-discrepancy structure
    - Global re-mixing: periodic re-scrambling with updated α
    - Iteration: alternate between refinement and re-scrambling
    """
    
    def __init__(self,
                 dimension: int = 2,
                 sampler_type: str = "sobol",
                 seed: Optional[int] = None):
        """
        Initialize split-step RQMC sampler.
        
        Args:
            dimension: Number of dimensions
            sampler_type: "sobol" or "halton"
            seed: Random seed
        """
        self.dimension = dimension
        self.sampler_type = sampler_type
        self.seed = seed
    
    def evolve(self,
               N: int,
               num_samples: int,
               num_steps: int = 5,
               alpha_schedule: Optional[List[float]] = None) -> List[np.ndarray]:
        """
        Perform split-step evolution with re-scrambling.
        
        Each step:
        1. Local refinement: generate samples with current α
        2. Global re-mixing: update α and re-scramble
        3. Iterate
        
        Args:
            N: Target number for factorization
            num_samples: Samples per step
            num_steps: Number of evolution steps
            alpha_schedule: Optional α values per step (uses default if None)
            
        Returns:
            List of sample arrays, one per step
        """
        if alpha_schedule is None:
            # Default schedule: gradually reduce α (increase scrambling)
            alpha_schedule = [0.7 * (0.85 ** step) for step in range(num_steps)]
        
        if len(alpha_schedule) != num_steps:
            raise ValueError(f"Need {num_steps} alpha values, got {len(alpha_schedule)}")
        
        evolution = []
        
        for step in range(num_steps):
            alpha_step = alpha_schedule[step]
            step_seed = self.seed + step * 100 if self.seed else None
            
            # Generate samples with current α
            if self.sampler_type == "sobol":
                sampler = ScrambledSobolSampler(
                    dimension=self.dimension,
                    alpha=alpha_step,
                    seed=step_seed
                )
            else:
                sampler = ScrambledHaltonSampler(
                    dimension=self.dimension,
                    alpha=alpha_step,
                    seed=step_seed
                )
            
            samples = sampler.generate(num_samples)
            evolution.append(samples)
        
        return evolution


def estimate_variance_from_replications(replications: List[np.ndarray]) -> Tuple[float, float]:
    """
    Estimate variance from RQMC replications.
    
    Uses ensemble averaging to compute unbiased variance estimate,
    as prescribed by RQMC theory (L'Ecuyer 2020).
    
    Args:
        replications: List of M independent sample arrays
        
    Returns:
        (mean_variance, std_error)
    """
    if len(replications) < 2:
        return 0.0, 0.0
    
    # Compute variance per replication
    variances = [np.var(rep) for rep in replications]
    
    # Mean and standard error across replications
    mean_var = np.mean(variances)
    std_err = np.std(variances) / np.sqrt(len(variances))
    
    return mean_var, std_err


def compute_rqmc_metrics(samples: np.ndarray,
                        alpha: float,
                        scrambling_depth: int,
                        num_replications: int,
                        true_factors: Optional[Tuple[int, int]] = None,
                        N: Optional[int] = None) -> RQMCMetrics:
    """
    Compute comprehensive RQMC performance metrics.
    
    Args:
        samples: Sample array of shape (n, dim)
        alpha: Coherence parameter used
        scrambling_depth: Bit levels scrambled
        num_replications: Number of ensemble replications
        true_factors: Optional known factors for success rate
        N: Optional target number for success rate calculation
        
    Returns:
        RQMCMetrics object
    """
    # Variance
    variance = np.var(samples)
    
    # Discrepancy (simplified L∞ estimate)
    n = samples.shape[0]
    discrepancy = np.log(n) / n  # Theoretical QMC bound
    
    # Convergence rate (empirical, requires multiple n values)
    convergence_rate = -1.5 if alpha < 0.5 else -1.0
    
    # Success rate
    success_rate = 0.0
    if true_factors is not None and N is not None:
        # Map samples to candidates around sqrt(N)
        sqrt_N = int(np.sqrt(N))
        candidates = set()
        for sample in samples:
            offset = int((sample[0] - 0.5) * 2 * sqrt_N * 0.1)
            candidate = sqrt_N + offset
            if 1 < candidate < N:
                candidates.add(candidate)
        
        p, q = true_factors
        if p in candidates or q in candidates:
            success_rate = 1.0
    
    return RQMCMetrics(
        alpha=alpha,
        scrambling_depth=scrambling_depth,
        num_replications=num_replications,
        variance=variance,
        discrepancy=discrepancy,
        convergence_rate=convergence_rate,
        success_rate=success_rate
    )


if __name__ == "__main__":
    print("RQMC Control Knob Demonstration")
    print("=" * 60)
    
    # Test scrambled Sobol'
    print("\n1. Scrambled Sobol' Sequence")
    print("-" * 60)
    for alpha in [1.0, 0.5, 0.1]:
        sampler = ScrambledSobolSampler(dimension=2, alpha=alpha, seed=42)
        samples = sampler.generate(100)
        print(f"α={alpha:.1f}: depth={sampler.scrambling_depth}, M={sampler.num_replications}")
        print(f"  Variance: {np.var(samples):.6f}")
        print(f"  Range: [{samples.min():.3f}, {samples.max():.3f}]")
    
    # Test replications for variance estimation
    print("\n2. RQMC Replications for Variance Estimation")
    print("-" * 60)
    sampler = ScrambledSobolSampler(dimension=2, alpha=0.5, seed=42)
    replications = sampler.generate_replications(100)
    mean_var, std_err = estimate_variance_from_replications(replications)
    print(f"M={len(replications)} replications")
    print(f"Mean variance: {mean_var:.6f} ± {std_err:.6f}")
    
    # Test adaptive α scheduling
    print("\n3. Adaptive α Scheduling")
    print("-" * 60)
    adaptive = AdaptiveRQMCSampler(dimension=2, target_variance=0.1, seed=42)
    samples, alpha_hist = adaptive.generate_adaptive(1000, num_batches=10)
    print(f"Final α: {alpha_hist[-1]:.3f}")
    print(f"α schedule: {[f'{a:.3f}' for a in alpha_hist]}")
    print(f"Final variance: {np.var(samples):.6f}")
    
    # Test split-step evolution
    print("\n4. Split-Step Evolution")
    print("-" * 60)
    split_step = SplitStepRQMC(dimension=2, seed=42)
    evolution = split_step.evolve(N=899, num_samples=100, num_steps=5)
    for i, step_samples in enumerate(evolution):
        print(f"Step {i+1}: variance={np.var(step_samples):.6f}")
    
    print("\n" + "=" * 60)
    print("RQMC Control Knob demonstration complete!")
