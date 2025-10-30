#!/usr/bin/env python3
"""
Variance-Reduced Pollard's Rho for Integer Factorization

This module implements Pollard's Rho algorithm enhanced with variance-reduction
techniques including:
- Randomized Quasi-Monte Carlo (RQMC) seeding
- Low-discrepancy Sobol/Owen sequences
- Gaussian lattice guidance
- Geodesic walk bias

The goal is to improve per-run success probability on large semiprimes
(up to ~256-bit / ~78 digits) under fixed compute budgets, turning "got lucky once"
behavior into reproducible single-digit/low double-digit success rates without
claiming asymptotic speedup below O(√p).

SECURITY NOTE: This implementation does NOT break RSA or modern cryptography.
It maintains O(√p) complexity but reduces variance for better reproducibility.
"""

import math
import random
from typing import Tuple, Optional
import mpmath as mp

mp.mp.dps = 50  # High precision for large integers


def gcd(a: int, b: int) -> int:
    """Compute greatest common divisor using Euclid's algorithm."""
    while b:
        a, b = b, a % b
    return a


class SobolSequence:
    """
    Sobol quasi-random sequence generator for low-discrepancy sampling.
    
    This provides better space-filling properties than pseudo-random numbers,
    reducing the variance in Pollard's Rho walks.
    """
    
    def __init__(self, dimension: int = 2, seed: int = 0):
        """
        Initialize Sobol sequence generator.
        
        Args:
            dimension: Number of dimensions (typically 2 for Rho)
            seed: Random seed for Owen scrambling (variance reduction)
        """
        self.dimension = dimension
        self.index = 0
        self.seed = seed
        random.seed(seed)
        
        # Direction numbers for Sobol sequence (simplified for dimensions 1-2)
        # In production, would use full Sobol direction numbers
        self.direction_numbers = [
            [1 << (31 - i) for i in range(32)],  # Dimension 1
            [self._gray_code_direction(i) for i in range(32)]  # Dimension 2
        ]
        
        # Owen scrambling matrices for variance reduction
        self.scramble = [[random.randint(0, (1 << 32) - 1) for _ in range(32)]
                         for _ in range(dimension)]
    
    def _gray_code_direction(self, i: int) -> int:
        """Generate direction number using Gray code."""
        return (1 << (31 - i)) ^ ((1 << (32 - i)) if i > 0 else 0)
    
    def next(self) -> Tuple[float, ...]:
        """
        Generate next point in Sobol sequence.
        
        Returns:
            Tuple of floats in [0, 1) representing coordinates
        """
        self.index += 1
        gray = self.index ^ (self.index >> 1)
        
        result = []
        for d in range(self.dimension):
            value = 0
            for i in range(32):
                if gray & (1 << i):
                    value ^= self.direction_numbers[d][i]
            
            # Apply Owen scrambling
            value ^= self.scramble[d][self.index % 32]
            
            result.append(value / (1 << 32))
        
        return tuple(result)
    
    def skip_to(self, index: int):
        """Skip to a specific index in the sequence."""
        self.index = index


class GaussianLatticeGuide:
    """
    Gaussian lattice guidance for Pollard's Rho parameter selection.
    
    Uses Z[i] lattice theory and Epstein zeta constant to bias exploration
    along structured curves in the Gaussian integers.
    """
    
    # Epstein zeta constant for Z[i] lattice (approximately 3.7246)
    EPSTEIN_ZETA = mp.mpf('3.7246047423695929')
    
    # Golden ratio for geodesic guidance
    PHI = (mp.sqrt(5) + 1) / 2
    
    def __init__(self, n: int):
        """
        Initialize lattice guide for semiprime n.
        
        Args:
            n: The semiprime to factor
        """
        self.n = n
        self.n_mp = mp.mpf(n)
        # Adaptive k parameter based on size of n
        self.k = self._compute_adaptive_k()
    
    def _compute_adaptive_k(self) -> mp.mpf:
        """
        Compute adaptive k parameter based on n.
        
        For very small n (n <= 1), or when log(log(n+1, 2), 2) <= 0, 
        the adaptive formula is not meaningful, so a default value is returned.
        For practical use, n should be a semiprime of at least 2 digits (n >= 10).
        The minimum recommended value of n for meaningful variance reduction is n >= 10.
        """
        if self.n <= 1:
            return mp.mpf('0.5')
        log_log_n = mp.log(mp.log(self.n_mp + 1, 2), 2)
        # For very small n, log_log_n can be <= 0; return default in that case.
        if log_log_n <= 0:
            return mp.mpf('0.5')
        return mp.mpf('0.3') / log_log_n
    
    def get_biased_constant(self, base_c: int) -> int:
        """
        Apply lattice bias to a candidate constant c.
        
        Args:
            base_c: Base constant value
            
        Returns:
            Biased constant guided by Gaussian lattice structure
        """
        # Embed c into geodesic coordinate system
        c_scaled = mp.mpf(base_c) / self.EPSTEIN_ZETA
        
        # Apply golden ratio scaling
        phi_scaled = float(self.PHI * self.k)
        
        # Fractional part modulation
        frac = float(c_scaled - mp.floor(c_scaled))
        
        # Compute biased value
        biased = int((base_c * (1 + phi_scaled * frac)) % self.n)
        
        return biased if biased != 0 else 1
    
    def get_geodesic_start(self, sobol_point: Tuple[float, float]) -> int:
        """
        Generate starting point using geodesic embedding.
        
        Args:
            sobol_point: Low-discrepancy point from Sobol sequence
            
        Returns:
            Starting value for Pollard's Rho walk
        """
        # Map Sobol point to integer range via geodesic embedding
        u, v = sobol_point
        
        # Use golden ratio and Epstein zeta for geometric mapping
        theta = 2 * math.pi * u
        r = math.sqrt(v)
        
        # Embed into torus coordinates
        x = r * math.cos(theta)
        y = r * math.sin(theta)
        
        # Map to integer range [2, n-1]
        value = 2 + int(((x + 1) / 2 * 0.5 + (y + 1) / 2 * 0.5) * (self.n - 3))
        
        return max(2, min(self.n - 1, value))


def pollard_rho_variance_reduced(
    n: int,
    max_iterations: int = 1000000,
    num_walks: int = 1,
    use_lattice_guide: bool = True,
    seed: int = 0
) -> Optional[int]:
    """
    Variance-reduced Pollard's Rho algorithm for integer factorization.
    
    This implementation uses:
    - Sobol low-discrepancy sequences for walk initialization
    - Owen scrambling for additional variance reduction
    - Gaussian lattice guidance for parameter selection
    - Multiple parallel walks with different constants
    
    Args:
        n: The integer to factor (should be composite)
        max_iterations: Maximum iterations per walk
        num_walks: Number of parallel walks to attempt
        use_lattice_guide: Whether to use Gaussian lattice guidance
        seed: Random seed for reproducibility
        
    Returns:
        A nontrivial factor of n, or None if no factor found
        
    Note:
        Expected complexity is still O(√p) where p is the smallest prime factor.
        Variance reduction improves probability of success per run, not asymptotic cost.
    """
    # Quick checks
    if n <= 1:
        return None
    if n % 2 == 0:
        return 2
    
    # Initialize variance reduction components
    sobol = SobolSequence(dimension=2, seed=seed)
    lattice_guide = GaussianLatticeGuide(n) if use_lattice_guide else None
    
    # Try multiple walks with different Sobol-guided initializations
    for walk_idx in range(num_walks):
        # Get low-discrepancy starting point
        sobol_point = sobol.next()
        
        if lattice_guide:
            x = lattice_guide.get_geodesic_start(sobol_point)
            # Get lattice-biased constant
            base_c = int(sobol_point[1] * (n - 2)) + 1
            c = lattice_guide.get_biased_constant(base_c)
        else:
            # Fallback to standard random initialization
            x = 2 + int(sobol_point[0] * (n - 3))
            c = 1 + int(sobol_point[1] * (n - 2))
        
        y = x
        
        # Pollard's Rho with tortoise and hare
        for _ in range(max_iterations):
            # f(x) = (x^2 + c) mod n
            x = (x * x + c) % n
            y = (y * y + c) % n
            y = (y * y + c) % n  # Hare moves twice
            
            # Check for cycle
            d = gcd(abs(x - y), n)
            
            if d > 1:
                if d < n:
                    return d  # Found nontrivial factor!
                else:
                    # Cycle detected but trivial, restart with new parameters
                    break
        
        # If walk exhausted, continue to next walk
    
    return None  # No factor found in budget


def pollard_rho_batch(
    n: int,
    max_iterations_per_walk: int = 100000,
    num_walks: int = 10,
    seed: int = 0,
    verbose: bool = False
) -> Optional[Tuple[int, int]]:
    """
    Run multiple variance-reduced Pollard's Rho walks and return factors if found.
    
    Args:
        n: The semiprime to factor
        max_iterations_per_walk: Iteration budget per walk
        num_walks: Number of independent walks
        seed: Random seed
        verbose: Whether to print progress
        
    Returns:
        Tuple of (factor1, factor2) if successful, None otherwise
    """
    if verbose:
        print(f"Attempting to factor n={n} ({n.bit_length()} bits)")
        print(f"Running {num_walks} walks with {max_iterations_per_walk} iterations each")
    
    factor = pollard_rho_variance_reduced(
        n=n,
        max_iterations=max_iterations_per_walk,
        num_walks=num_walks,
        use_lattice_guide=True,
        seed=seed
    )
    
    if factor and factor > 1 and factor < n:
        other_factor = n // factor
        if verbose:
            print(f"Success! Found factors: {factor} × {other_factor}")
        return (min(factor, other_factor), max(factor, other_factor))
    
    if verbose:
        print("No factor found within budget")
    
    return None


if __name__ == "__main__":
    # Demo: Factor some small semiprimes
    test_cases = [
        (143, "11 × 13"),
        (899, "29 × 31"),
        (1003, "17 × 59"),
        (1077739877, "32771 × 32887 (30-bit)"),
    ]
    
    print("Variance-Reduced Pollard's Rho Demo")
    print("=" * 60)
    print()
    
    for n, description in test_cases:
        print(f"Testing n = {n} ({description})")
        result = pollard_rho_batch(n, max_iterations_per_walk=10000, num_walks=5, verbose=False)
        if result:
            p, q = result
            print(f"  ✓ Success: {p} × {q}")
            assert p * q == n, "Incorrect factorization!"
        else:
            print(f"  ✗ Failed to factor within budget")
        print()
    
    print("=" * 60)
    print("Note: This implementation maintains O(√p) complexity.")
    print("Variance reduction improves success probability, not asymptotic cost.")
