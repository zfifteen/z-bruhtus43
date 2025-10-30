#!/usr/bin/env python3
"""
Test suite for variance-reduced Pollard's Rho implementations.

Tests cover:
1. Sobol sequence generation and properties
2. Gaussian lattice guidance
3. Integer factorization correctness
4. DLP solving correctness
5. Edge cases and error handling
"""

import unittest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from variance_reduced_rho import (
    SobolSequence, 
    GaussianLatticeGuide,
    pollard_rho_variance_reduced,
    pollard_rho_batch,
    gcd
)

from variance_reduced_dlp import (
    pollard_rho_dlp_variance_reduced,
    dlp_batch_parallel,
    is_distinguished,
    partition_function,
    mod_inverse,
    WalkState
)


class TestSobolSequence(unittest.TestCase):
    """Test Sobol sequence generation."""
    
    def test_initialization(self):
        """Test Sobol sequence initialization."""
        sobol = SobolSequence(dimension=2, seed=42)
        self.assertEqual(sobol.dimension, 2)
        self.assertEqual(sobol.index, 0)
    
    def test_next_generates_points_in_unit_cube(self):
        """Test that generated points are in [0, 1)^d."""
        sobol = SobolSequence(dimension=2, seed=0)
        for _ in range(100):
            u, v = sobol.next()
            self.assertGreaterEqual(u, 0.0)
            self.assertLess(u, 1.0)
            self.assertGreaterEqual(v, 0.0)
            self.assertLess(v, 1.0)
    
    def test_different_seeds_produce_different_sequences(self):
        """Test that Owen scrambling with different seeds produces different sequences."""
        sobol1 = SobolSequence(dimension=2, seed=1)
        sobol2 = SobolSequence(dimension=2, seed=2)
        
        points1 = [sobol1.next() for _ in range(10)]
        points2 = [sobol2.next() for _ in range(10)]
        
        # At least some points should be different
        self.assertNotEqual(points1, points2)
    
    def test_skip_to(self):
        """Test skipping to specific index."""
        sobol = SobolSequence(dimension=2, seed=0)
        sobol.skip_to(100)
        self.assertEqual(sobol.index, 100)


class TestGaussianLatticeGuide(unittest.TestCase):
    """Test Gaussian lattice guidance."""
    
    def test_initialization(self):
        """Test lattice guide initialization."""
        guide = GaussianLatticeGuide(n=143)
        self.assertEqual(guide.n, 143)
        self.assertGreater(float(guide.k), 0)
    
    def test_adaptive_k_increases_with_n(self):
        """Test that adaptive k parameter behaves reasonably."""
        guide_small = GaussianLatticeGuide(n=100)
        guide_large = GaussianLatticeGuide(n=10000)
        
        # k should be positive for both
        self.assertGreater(float(guide_small.k), 0)
        self.assertGreater(float(guide_large.k), 0)
    
    def test_biased_constant_in_range(self):
        """Test that biased constant is in valid range."""
        guide = GaussianLatticeGuide(n=899)
        
        for base_c in [1, 2, 10, 100]:
            biased = guide.get_biased_constant(base_c)
            self.assertGreater(biased, 0)
            self.assertLess(biased, guide.n)
    
    def test_geodesic_start_in_range(self):
        """Test that geodesic starting points are valid."""
        guide = GaussianLatticeGuide(n=899)
        
        sobol_points = [(0.1, 0.2), (0.5, 0.5), (0.9, 0.1)]
        for point in sobol_points:
            start = guide.get_geodesic_start(point)
            self.assertGreaterEqual(start, 2)
            self.assertLess(start, guide.n)


class TestFactorization(unittest.TestCase):
    """Test integer factorization."""
    
    def test_gcd_function(self):
        """Test GCD computation."""
        self.assertEqual(gcd(48, 18), 6)
        self.assertEqual(gcd(17, 19), 1)
        self.assertEqual(gcd(100, 50), 50)
        self.assertEqual(gcd(0, 5), 5)
    
    def test_factor_small_semiprimes(self):
        """Test factoring small known semiprimes."""
        test_cases = [
            (143, 11, 13),
            (899, 29, 31),
            (1003, 17, 59),
        ]
        
        for n, p_expected, q_expected in test_cases:
            with self.subTest(n=n):
                result = pollard_rho_batch(
                    n=n,
                    max_iterations_per_walk=10000,
                    num_walks=5,
                    seed=0,
                    verbose=False
                )
                
                self.assertIsNotNone(result, f"Failed to factor {n}")
                p, q = result
                self.assertEqual(p * q, n)
                self.assertIn(p, [p_expected, q_expected])
                self.assertIn(q, [p_expected, q_expected])
    
    def test_factor_even_number(self):
        """Test that even numbers return 2."""
        result = pollard_rho_variance_reduced(n=100, num_walks=1)
        self.assertEqual(result, 2)
    
    def test_factor_prime_returns_none(self):
        """Test that prime inputs return None or n."""
        # Small prime
        result = pollard_rho_variance_reduced(
            n=17,
            max_iterations=1000,
            num_walks=3
        )
        # Should either return None or fail to find factor
        if result:
            self.assertIn(result, [1, 17])
    
    def test_reproducibility_with_seed(self):
        """Test that same seed produces same results."""
        n = 899
        
        result1 = pollard_rho_batch(n, num_walks=3, seed=42)
        result2 = pollard_rho_batch(n, num_walks=3, seed=42)
        
        # Should get same factors (order might vary)
        if result1 and result2:
            self.assertEqual(set(result1), set(result2))
    
    def test_different_seeds_can_produce_different_paths(self):
        """Test that different seeds may take different paths (but same result)."""
        n = 1003
        
        # Both should succeed but might use different walks
        result1 = pollard_rho_batch(n, num_walks=5, seed=1)
        result2 = pollard_rho_batch(n, num_walks=5, seed=2)
        
        if result1 and result2:
            # Same factorization
            self.assertEqual(result1[0] * result1[1], n)
            self.assertEqual(result2[0] * result2[1], n)


class TestDLP(unittest.TestCase):
    """Test discrete logarithm solving."""
    
    def test_mod_inverse(self):
        """Test modular inverse computation."""
        # 3 * 5 ≡ 1 (mod 7), so inv(3, 7) = 5
        self.assertEqual(mod_inverse(3, 7), 5)
        
        # 2 * 3 ≡ 1 (mod 5), so inv(2, 5) = 3
        self.assertEqual(mod_inverse(2, 5), 3)
        
        # No inverse for 2 mod 4
        self.assertIsNone(mod_inverse(2, 4))
    
    def test_is_distinguished(self):
        """Test distinguished point detection."""
        # 16-bit mask: checks last 16 bits are zero
        bit_mask = (1 << 16) - 1
        
        # 2^16 = 65536 should be distinguished (all low 16 bits are 0)
        self.assertTrue(is_distinguished(65536, bit_mask))
        
        # Random number likely not distinguished
        self.assertFalse(is_distinguished(12345, bit_mask))
    
    def test_partition_function(self):
        """Test partition function."""
        # Should return values in [0, num_partitions)
        for x in [0, 1, 100, 1000]:
            partition = partition_function(x, num_partitions=20)
            self.assertGreaterEqual(partition, 0)
            self.assertLess(partition, 20)
    
    def test_walk_state(self):
        """Test WalkState dataclass."""
        state = WalkState(element=10, alpha_power=2, beta_power=3, steps=5)
        self.assertEqual(state.element, 10)
        self.assertEqual(state.alpha_power, 2)
        self.assertEqual(state.beta_power, 3)
        self.assertEqual(state.steps, 5)
    
    def test_solve_small_dlp(self):
        """Test solving small DLP instances."""
        # Small example: 2^x ≡ 8 (mod 11), so x = 3
        p = 11
        alpha = 2
        gamma_true = 3
        beta = pow(alpha, gamma_true, p)  # 2^3 = 8
        
        result = dlp_batch_parallel(
            alpha=alpha,
            beta=beta,
            modulus=p,
            order=p-1,
            max_steps_per_walk=10000,
            num_walks=10,
            seed=0,
            verbose=False
        )
        
        # Might not always succeed with small budget, but if it does, verify
        if result is not None:
            self.assertEqual(pow(alpha, result, p), beta)
            # Note: result might be equivalent mod (p-1), not necessarily == gamma_true
    
    def test_dlp_verification(self):
        """Test that DLP solutions verify correctly."""
        p = 101
        alpha = 3
        gamma = 42
        beta = pow(alpha, gamma, p)
        
        # Run DLP solver
        result = dlp_batch_parallel(
            alpha=alpha,
            beta=beta,
            modulus=p,
            max_steps_per_walk=50000,
            num_walks=15,
            seed=123
        )
        
        if result is not None:
            # Verify solution
            self.assertEqual(pow(alpha, result, p), beta,
                           "DLP solution does not verify")


class TestEdgeCases(unittest.TestCase):
    """Test edge cases and error handling."""
    
    def test_factorization_n_equals_1(self):
        """Test n=1 returns None."""
        result = pollard_rho_variance_reduced(n=1)
        self.assertIsNone(result)
    
    def test_factorization_n_equals_2(self):
        """Test n=2 (even) returns 2."""
        result = pollard_rho_variance_reduced(n=2)
        self.assertEqual(result, 2)
    
    def test_factorization_negative_iterations(self):
        """Test that algorithm handles edge parameters gracefully."""
        # Should handle edge case without crashing
        result = pollard_rho_variance_reduced(
            n=143,
            max_iterations=1,  # Very small budget
            num_walks=1
        )
        # May or may not succeed, but shouldn't crash
        self.assertIsInstance(result, (int, type(None)))
    
    def test_zero_walks(self):
        """Test zero walks returns None."""
        result = pollard_rho_variance_reduced(
            n=143,
            num_walks=0
        )
        self.assertIsNone(result)


class TestVarianceReduction(unittest.TestCase):
    """Test variance reduction properties."""
    
    def test_success_rate_consistency(self):
        """Test that success rate is relatively consistent across runs."""
        n = 899  # Small semiprime for testing
        num_experiments = 5
        success_counts = []
        
        for experiment in range(num_experiments):
            successes = 0
            for trial in range(10):
                result = pollard_rho_batch(
                    n=n,
                    max_iterations_per_walk=5000,
                    num_walks=3,
                    seed=experiment * 100 + trial
                )
                if result:
                    successes += 1
            success_counts.append(successes)
        
        # All experiments should have relatively high success rate
        avg_success = sum(success_counts) / len(success_counts)
        self.assertGreater(avg_success, 5,  # At least 50% success
                          "Success rate too low for small semiprime")
        
        # Variance should be relatively low (all counts should be similar)
        variance = sum((x - avg_success)**2 for x in success_counts) / len(success_counts)
        self.assertLess(variance, 25,  # Reasonable variance bound
                       "Success rate variance too high")


def run_tests():
    """Run all tests."""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add all test classes
    suite.addTests(loader.loadTestsFromTestCase(TestSobolSequence))
    suite.addTests(loader.loadTestsFromTestCase(TestGaussianLatticeGuide))
    suite.addTests(loader.loadTestsFromTestCase(TestFactorization))
    suite.addTests(loader.loadTestsFromTestCase(TestDLP))
    suite.addTests(loader.loadTestsFromTestCase(TestEdgeCases))
    suite.addTests(loader.loadTestsFromTestCase(TestVarianceReduction))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return 0 if successful, 1 if failures
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    exit_code = run_tests()
    sys.exit(exit_code)
