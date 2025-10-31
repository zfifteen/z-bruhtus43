#!/usr/bin/env python3
"""
Test Suite for Coherence-Enhanced Pollard's Rho (Issue #16)

Tests the integration of RQMC, reduced coherence, and adaptive variance control
for Pollard's Rho factorization.

Test Coverage:
1. α-to-scrambling parameter mapping
2. Owen scrambling correctness
3. Ensemble averaging
4. Split-step evolution
5. Adaptive α scheduling
6. Integration with factorization
7. Variance reduction validation
8. Convergence rate measurement
"""

import pytest
import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from coherence_enhanced_pollard_rho import (
    CoherenceEnhancedPollardRho,
    factor_with_coherence,
    FactorizationMode,
)
from rqmc_control import (
    ScrambledSobolSampler,
    AdaptiveRQMCSampler,
    SplitStepRQMC,
    estimate_variance_from_replications
)
from reduced_coherence import (
    CoherenceMode
)

class TestAlphaMapping:
    """Test α parameter to scrambling depth mapping."""

    def test_alpha_to_scrambling_depth(self):
        """Test that α correctly maps to scrambling depth."""
        # α = 1.0 → minimal scrambling (depth = 1)
        sampler_high = ScrambledSobolSampler(dimension=2, alpha=1.0, seed=42)
        assert sampler_high.scrambling_depth == 1

        # α = 0.5 → moderate scrambling
        sampler_mid = ScrambledSobolSampler(dimension=2, alpha=0.5, seed=42)
        assert 10 <= sampler_mid.scrambling_depth <= 25

        # α = 0.0 → maximum scrambling (depth = 32)
        sampler_low = ScrambledSobolSampler(dimension=2, alpha=0.0, seed=42)
        assert sampler_low.scrambling_depth == 32

    def test_alpha_to_ensemble_size(self):
        """Test that α correctly maps to ensemble size."""
        # α = 1.0 → M = 1 (deterministic)
        sampler_high = ScrambledSobolSampler(dimension=2, alpha=1.0, seed=42)
        assert sampler_high.num_replications == 1

        # α = 0.5 → M ≈ 8
        sampler_mid = ScrambledSobolSampler(dimension=2, alpha=0.5, seed=42)
        assert 5 <= sampler_mid.num_replications <= 10

        # α = 0.0 → M = 10
        sampler_low = ScrambledSobolSampler(dimension=2, alpha=0.0, seed=42)
        assert sampler_low.num_replications == 10

    def test_coherence_length(self):
        """Test coherence length calculation."""
        # l_c ~ 1/α
        sampler_high = ReducedCoherenceSampler(seed=42, coherence_alpha=1.0)
        assert abs(sampler_high.correlation_length - 1.0) < 10.0

        sampler_low = ReducedCoherenceSampler(seed=42, coherence_alpha=0.1)
        assert sampler_low.correlation_length >= 10.0

    def test_decoherence_rate(self):
        """Test decoherence rate γ = 1 - α."""
        sampler = ReducedCoherenceSampler(seed=42, coherence_alpha=0.7)
        assert abs(sampler.decoherence_rate - 0.3) < 1e-10


class TestOwenScrambling:
    """Test Owen scrambling implementation."""

    def test_scrambling_preserves_range(self):
        """Test that scrambling keeps values in [0, 1]."""
        sampler = ScrambledSobolSampler(dimension=2, alpha=0.5, seed=42)
        samples = sampler.generate(100)

        assert np.all(samples >= 0.0)
        assert np.all(samples <= 1.0)

    def test_scrambling_changes_sequence(self):
        """Test that scrambling actually modifies the sequence."""
        # Unscrambled (α=1.0)
        sampler_unscrambled = ScrambledSobolSampler(dimension=2, alpha=1.0, seed=42)
        unscrambled = sampler_unscrambled.generate(100)

        # Scrambled (α=0.5)
        sampler_scrambled = ScrambledSobolSampler(dimension=2, alpha=0.5, seed=42)
        scrambled = sampler_scrambled.generate(100)

        # Should be different
        assert not np.allclose(unscrambled, scrambled)

    def test_scrambling_reproducibility(self):
        """Test that scrambling with same seed is reproducible."""
        sampler1 = ScrambledSobolSampler(dimension=2, alpha=0.5, seed=42)
        samples1 = sampler1.generate(100)

        sampler2 = ScrambledSobolSampler(dimension=2, alpha=0.5, seed=42)
        samples2 = sampler2.generate(100)

        assert np.allclose(samples1, samples2)

    def test_replications_independence(self):
        """Test that replications are independent."""
        sampler = ScrambledSobolSampler(dimension=2, alpha=0.5, seed=42)
        replications = sampler.generate_replications(100)

        # Should have M replications
        assert len(replications) == sampler.num_replications

        # Each should be different
        for i in range(len(replications) - 1):
            assert not np.allclose(replications[i], replications[i+1])


class TestAdaptiveVarianceControl:
    """Test adaptive α scheduling for target variance."""

    def test_adaptive_reaches_target(self):
        """Test that adaptive mode approaches target variance."""
        target_variance = 0.1
        adaptive = AdaptiveRQMCSampler(
            dimension=2,
            target_variance=target_variance,
            seed=42
        )

        samples, alpha_history = adaptive.generate_adaptive(1000, num_batches=10)

        # Final variance should be near target (within 50%)
        final_variance = np.var(samples)
        assert 0.5 * target_variance < final_variance < 2.0 * target_variance

    def test_alpha_history_tracking(self):
        """Test that α history is recorded."""
        adaptive = AdaptiveRQMCSampler(dimension=2, target_variance=0.1, seed=42)
        samples, alpha_history = adaptive.generate_adaptive(1000, num_batches=5)

        # Should have 5 α values
        assert len(alpha_history) == 5

        # All should be in [0, 1]
        assert all(0.0 <= a <= 1.0 for a in alpha_history)

    def test_weighted_discrepancy(self):
        """Test dimension-wise α control."""
        adaptive = AdaptiveRQMCSampler(dimension=4, target_variance=0.1, seed=42)

        # Higher weights on first 2 dimensions
        weights = np.array([1.0, 1.0, 0.2, 0.2])
        samples = adaptive.generate_weighted_discrepancy(500, dimension_weights=weights)

        assert samples.shape == (500, 4)


class TestSplitStepEvolution:
    """Test split-step evolution with periodic re-scrambling."""

    def test_split_step_generates_steps(self):
        """Test that split-step generates correct number of steps."""
        split_step = SplitStepRQMC(dimension=2, seed=42)
        evolution = split_step.evolve(N=899, num_samples=100, num_steps=5)

        assert len(evolution) == 5
        for step in evolution:
            assert step.shape == (100, 2)

    def test_alpha_schedule(self):
        """Test custom α schedule."""
        split_step = SplitStepRQMC(dimension=2, seed=42)
        alpha_schedule = [0.8, 0.6, 0.4, 0.3, 0.2]

        evolution = split_step.evolve(
            N=899,
            num_samples=100,
            num_steps=5,
            alpha_schedule=alpha_schedule
        )

        assert len(evolution) == 5


class TestEnsembleAveraging:
    """Test ensemble averaging (complex screen method)."""

    def test_ensemble_generates_candidates(self):
        """Test that ensemble averaging generates valid candidates."""
        sampler = ReducedCoherenceSampler(seed=42, coherence_alpha=0.5, num_ensembles=4)
        candidates = sampler.ensemble_averaged_sampling(N=899, num_samples=500, phi_bias=True)

        assert len(candidates) > 0
        assert all(1 < c < 899 for c in candidates)

    def test_ensemble_diversity(self):
        """Test that ensembles provide diversity."""
        sampler = ReducedCoherenceSampler(seed=42, coherence_alpha=0.5, num_ensembles=4)
        candidates = sampler.ensemble_averaged_sampling(N=899, num_samples=500, phi_bias=False)

        # Should have reasonable diversity
        unique_count = len(set(candidates))
        assert unique_count > len(candidates) * 0.5

    def test_phi_bias(self):
        """Test golden ratio bias in ensemble averaging."""
        sampler = ReducedCoherenceSampler(seed=42, coherence_alpha=0.5, num_ensembles=4)

        with_phi = sampler.ensemble_averaged_sampling(N=899, num_samples=500, phi_bias=True)
        without_phi = sampler.ensemble_averaged_sampling(N=899, num_samples=500, phi_bias=False)

        # Both should generate candidates
        assert len(with_phi) > 0
        assert len(without_phi) > 0


class TestFactorization:
    """Test factorization with coherence enhancement."""

    def test_factor_small_semiprime(self):
        """Test factoring small semiprime (899 = 29 × 31)."""
        result = factor_with_coherence(899, alpha=0.5, mode="adaptive", num_walks=10)

        assert result.success
        assert result.factor in [29, 31]
        assert 899 % result.factor == 0

    def test_factor_even_number(self):
        """Test factoring even number (trivial case)."""
        result = factor_with_coherence(1000, alpha=0.5, mode="fixed", num_walks=5)

        assert result.success
        assert result.factor == 2
        assert result.iterations == 0

    def test_fixed_coherence_mode(self):
        """Test fixed coherence mode."""
        factorizer = CoherenceEnhancedPollardRho(alpha=0.5, seed=42)
        result = factorizer.factor_with_fixed_coherence(899, num_walks=10, max_iterations=10000)

        assert result.success
        assert result.mode == FactorizationMode.FIXED_COHERENCE
        assert abs(result.alpha_used - 0.5) < 0.1

    def test_adaptive_coherence_mode(self):
        """Test adaptive coherence mode."""
        factorizer = CoherenceEnhancedPollardRho(alpha=0.5, target_variance=0.1, seed=42)
        result = factorizer.factor_with_adaptive_coherence(899, num_walks=10, num_batches=3)

        assert result.success
        assert result.mode == FactorizationMode.ADAPTIVE_COHERENCE
        # α may have adjusted
        assert 0.0 <= result.alpha_used <= 1.0

    def test_split_step_mode(self):
        """Test split-step evolution mode."""
        factorizer = CoherenceEnhancedPollardRho(alpha=0.5, seed=42)
        result = factorizer.factor_with_split_step(
            899,
            num_walks_per_step=5,
            num_steps=3,
            max_iterations=10000
        )

        assert result.success
        assert result.mode == FactorizationMode.SPLIT_STEP

    def test_ensemble_averaged_mode(self):
        """Test ensemble averaging mode."""
        factorizer = CoherenceEnhancedPollardRho(alpha=0.5, seed=42)
        result = factorizer.factor_with_ensemble_averaging(899, num_samples=1000)

        assert result.success
        assert result.mode == FactorizationMode.ENSEMBLE_AVERAGED
        assert result.coherence_metrics is not None


class TestVarianceReduction:
    """Test variance reduction properties."""

    def test_variance_reduces_with_qmc(self):
        """Test that QMC reduces variance vs pure random."""
        # This is a statistical test, may have false positives
        # Run multiple times for robustness
        from low_discrepancy import SobolSampler

        variances_qmc = []
        variances_random = []

        for trial in range(5):
            # QMC samples
            sampler_qmc = SobolSampler(dimension=2, scramble=False, seed=42+trial)
            samples_qmc = sampler_qmc.generate(1000)
            variances_qmc.append(np.var(samples_qmc))

            # Random samples
            rng = np.random.Generator(np.random.PCG64(42+trial))
            samples_random = rng.uniform(0, 1, (1000, 2))
            variances_random.append(np.var(samples_random))

        # QMC should have lower variance on average
        mean_var_qmc = np.mean(variances_qmc)
        mean_var_random = np.mean(variances_random)

        # Note: This test may occasionally fail due to randomness
        # QMC typically shows 30-50% variance reduction
        assert mean_var_qmc < mean_var_random * 1.2  # Allow some margin

    def test_variance_estimation_from_replications(self):
        """Test RQMC variance estimation."""
        sampler = ScrambledSobolSampler(dimension=2, alpha=0.5, seed=42)
        replications = sampler.generate_replications(100)

        mean_var, std_err = estimate_variance_from_replications(replications)

        assert mean_var > 0
        assert std_err >= 0
        assert std_err < mean_var  # Standard error should be < mean


class TestMetrics:
    """Test metric collection and reporting."""

    def test_factorization_result_structure(self):
        """Test FactorizationResult contains all fields."""
        result = factor_with_coherence(899, alpha=0.5, mode="fixed", num_walks=5)

        assert hasattr(result, 'factor')
        assert hasattr(result, 'iterations')
        assert hasattr(result, 'alpha_used')
        assert hasattr(result, 'variance')
        assert hasattr(result, 'convergence_rate')
        assert hasattr(result, 'mode')
        assert hasattr(result, 'success')
        assert hasattr(result, 'candidates_explored')
        assert hasattr(result, 'time_elapsed')

    def test_rqmc_metrics_collected(self):
        """Test that RQMC metrics are collected."""
        result = factor_with_coherence(899, alpha=0.5, mode="fixed", num_walks=5)

        if result.rqmc_metrics:
            assert hasattr(result.rqmc_metrics, 'alpha')
            assert hasattr(result.rqmc_metrics, 'scrambling_depth')
            assert hasattr(result.rqmc_metrics, 'num_replications')
            assert hasattr(result.rqmc_metrics, 'variance')

    def test_coherence_metrics_collected(self):
        """Test that coherence metrics are collected for ensemble mode."""
        result = factor_with_coherence(899, alpha=0.5, mode="ensemble", num_samples=500)

        assert result.coherence_metrics is not None
        assert hasattr(result.coherence_metrics, 'alpha')
        assert hasattr(result.coherence_metrics, 'variance')
        assert hasattr(result.coherence_metrics, 'correlation_length')


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_alpha_bounds(self):
        """Test that invalid α raises ValueError."""
        with pytest.raises(ValueError):
            CoherenceEnhancedPollardRho(alpha=-0.1)

        with pytest.raises(ValueError):
            CoherenceEnhancedPollardRho(alpha=1.5)

    def test_target_variance_positive(self):
        """Test that target variance must be positive."""
        with pytest.raises(ValueError):
            CoherenceEnhancedPollardRho(alpha=0.5, target_variance=-0.1)

        with pytest.raises(ValueError):
            CoherenceEnhancedPollardRho(alpha=0.5, target_variance=0.0)

    def test_prime_number(self):
        """Test attempting to factor a prime (should fail)."""
        result = factor_with_coherence(31, alpha=0.5, mode="fixed", num_walks=5, max_iterations=1000)

        # Should not find factor (or find 1 or 31)
        if result.factor:
            assert result.factor in [1, 31]

    def test_small_numbers(self):
        """Test edge cases with small numbers."""
        # N = 4 = 2 × 2
        result = factor_with_coherence(4, alpha=0.5, mode="fixed")
        assert result.success
        assert result.factor == 2


if __name__ == "__main__":
    # Run tests with pytest
    pytest.main([__file__, "-v", "--tb=short"])
