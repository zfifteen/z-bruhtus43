#!/usr/bin/env python3
"""
Variance-Reduced Pollard's Rho for Discrete Logarithm Problem (DLP)

This module implements Pollard's Rho for DLP enhanced with variance-reduction
techniques. The goal is to solve α^γ = β in a cyclic group G of order n.

Key features:
- RQMC seeding for walk initialization
- Low-discrepancy Sobol sequences for distinguished point walks
- Geometric biasing similar to integer factorization variant
- Distinguished point collision detection

SECURITY NOTE: This maintains O(√n) complexity for generic DLP. For a 256-bit
cyclic group, expected work is still ~2^128 operations. Variance reduction
improves probability of collision within fixed budget, not asymptotic cost.

This does NOT break modern ECC or cryptographic protocols.
"""

from typing import Optional, Tuple, List
from dataclasses import dataclass

# Import Sobol sequence from factorization module
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from variance_reduced_rho import SobolSequence


@dataclass
class WalkState:
    """State of a random walk in the group."""
    element: int  # Current group element
    alpha_power: int  # Exponent of α
    beta_power: int  # Exponent of β
    steps: int  # Number of steps taken


def is_distinguished(x: int, bit_mask: int) -> bool:
    """
    Check if a group element is a distinguished point.
    
    Distinguished points have a specific bit pattern (e.g., trailing zeros)
    to reduce memory requirements in parallel collision search.
    
    Args:
        x: Group element
        bit_mask: Bit mask defining distinguished point criterion
        
    Returns:
        True if x is a distinguished point
    """
    return (x & bit_mask) == 0


def partition_function(x: int, num_partitions: int = 20) -> int:
    """
    Partition function for Pollard's Rho walk in cyclic group.
    
    Divides group into regions to determine walk direction.
    Using more partitions (e.g., 20) reduces walk correlation.
    
    Args:
        x: Current group element
        num_partitions: Number of partitions
        
    Returns:
        Partition index in [0, num_partitions)
    """
    return x % num_partitions


def dlp_walk_step(
    state: WalkState,
    alpha: int,
    beta: int,
    modulus: int,
    partition_actions: List[Tuple[int, int]]
) -> WalkState:
    """
    Perform one step of Pollard's Rho walk for DLP.
    
    Walk is defined by partitions: multiply by α, multiply by β, or square,
    depending on the partition the current element falls into.
    
    Args:
        state: Current walk state
        alpha: Generator α
        beta: Target element β
        modulus: Group modulus (prime for Z_p^*)
        partition_actions: List of (alpha_increment, beta_increment) per partition
        
    Returns:
        New walk state
    """
    partition = partition_function(state.element, len(partition_actions))
    alpha_inc, beta_inc = partition_actions[partition]
    
    # Update element based on partition
    if alpha_inc == 1 and beta_inc == 0:
        # Multiply by α
        new_element = (state.element * alpha) % modulus
        new_alpha = state.alpha_power + 1
        new_beta = state.beta_power
    elif alpha_inc == 0 and beta_inc == 1:
        # Multiply by β
        new_element = (state.element * beta) % modulus
        new_alpha = state.alpha_power
        new_beta = state.beta_power + 1
    else:
        # Square (alpha_inc=2, beta_inc=2 in doubling)
        new_element = (state.element * state.element) % modulus
        new_alpha = (state.alpha_power * 2)
        new_beta = (state.beta_power * 2)
    
    return WalkState(
        element=new_element,
        alpha_power=new_alpha,
        beta_power=new_beta,
        steps=state.steps + 1
    )


def mod_inverse(a: int, m: int) -> Optional[int]:
    """
    Compute modular multiplicative inverse using extended Euclidean algorithm.
    
    Args:
        a: Number to invert
        m: Modulus
        
    Returns:
        Inverse of a modulo m, or None if it doesn't exist
    """
    def extended_gcd(a: int, b: int) -> Tuple[int, int, int]:
        if a == 0:
            return b, 0, 1
        gcd, x1, y1 = extended_gcd(b % a, a)
        x = y1 - (b // a) * x1
        y = x1
        return gcd, x, y
    
    gcd, x, _ = extended_gcd(a % m, m)
    if gcd != 1:
        return None
    return (x % m + m) % m


def pollard_rho_dlp_variance_reduced(
    alpha: int,
    beta: int,
    modulus: int,
    order: Optional[int] = None,
    max_steps: int = 1000000,
    num_walks: int = 10,
    distinguish_bits: int = 16,
    seed: int = 0,
    verbose: bool = False
) -> Optional[int]:
    """
    Variance-reduced Pollard's Rho for discrete logarithm problem.
    
    Solves α^γ ≡ β (mod modulus) for unknown exponent γ.
    
    Uses variance-reduction techniques:
    - Sobol sequences for walk initialization
    - Distinguished points for collision detection
    - Multiple parallel walks with low-discrepancy starts
    
    Args:
        alpha: Generator (base)
        beta: Target element
        modulus: Prime modulus defining cyclic group
        order: Group order (if known, otherwise use modulus-1)
        max_steps: Maximum steps per walk
        num_walks: Number of parallel walks to try
        distinguish_bits: Number of trailing zero bits for distinguished points
        seed: Random seed for reproducibility
        verbose: Print progress information
        
    Returns:
        The discrete logarithm γ such that α^γ ≡ β (mod modulus), or None
        
    Note:
        Expected complexity is O(√n) where n is the group order.
        For a 256-bit order group, this is ~2^128 operations.
        Variance reduction improves collision probability, not asymptotic cost.
    """
    if order is None:
        order = modulus - 1  # Assume Z_p^* for prime modulus
    
    # Initialize variance reduction
    sobol = SobolSequence(dimension=2, seed=seed)
    
    # Distinguished point bit mask
    bit_mask = (1 << distinguish_bits) - 1
    
    # Define partition actions: 1/3 multiply by α, 1/3 by β, 1/3 square
    num_partitions = 20
    partition_actions = []
    for i in range(num_partitions):
        if i < num_partitions // 3:
            partition_actions.append((1, 0))  # Multiply by α
        elif i < 2 * num_partitions // 3:
            partition_actions.append((0, 1))  # Multiply by β
        else:
            partition_actions.append((2, 2))  # Square (doubles exponents)
    
    # Storage for distinguished points
    distinguished_points = {}
    
    if verbose:
        print(f"DLP: Finding γ such that {alpha}^γ ≡ {beta} (mod {modulus})")
        print(f"Running {num_walks} walks with max {max_steps} steps each")
        print(f"Using {distinguish_bits}-bit distinguished points")
    
    # Run multiple walks with Sobol-guided initialization
    for walk_idx in range(num_walks):
        # Get low-discrepancy starting point
        sobol_point = sobol.next()
        
        # Initialize walk with geometric bias
        # Map Sobol point to exponent space
        start_alpha_exp = int(sobol_point[0] * order)
        start_beta_exp = int(sobol_point[1] * order)
        
        # Compute starting group element: α^a * β^b
        start_element = (pow(alpha, start_alpha_exp, modulus) *
                        pow(beta, start_beta_exp, modulus)) % modulus
        
        state = WalkState(
            element=start_element,
            alpha_power=start_alpha_exp,
            beta_power=start_beta_exp,
            steps=0
        )
        
        # Walk until distinguished point or max steps
        for _ in range(max_steps):
            state = dlp_walk_step(state, alpha, beta, modulus, partition_actions)
            
            # Check if distinguished
            if is_distinguished(state.element, bit_mask):
                # Check for collision
                if state.element in distinguished_points:
                    # Collision! Solve for discrete log
                    prev_state = distinguished_points[state.element]
                    
                    # We have: α^a1 * β^b1 ≡ α^a2 * β^b2 (mod modulus)
                    # This gives: α^(a1-a2) ≡ β^(b2-b1) (mod modulus)
                    # Since β = α^γ, we get: α^(a1-a2) ≡ α^(γ(b2-b1)) (mod modulus)
                    # Therefore: γ ≡ (a1-a2)/(b2-b1) (mod order)
                    
                    delta_alpha = (state.alpha_power - prev_state.alpha_power) % order
                    delta_beta = (prev_state.beta_power - state.beta_power) % order
                    
                    if delta_beta != 0:
                        # Compute modular inverse
                        inv = mod_inverse(delta_beta, order)
                        if inv is not None:
                            gamma = (delta_alpha * inv) % order
                            
                            # Verify solution
                            if pow(alpha, gamma, modulus) == beta:
                                if verbose:
                                    print(f"Success! Found γ = {gamma}")
                                    print(f"Collision after walk {walk_idx}, step {state.steps}")
                                return gamma
                
                # Store distinguished point
                distinguished_points[state.element] = state
    
    if verbose:
        print(f"No collision found after {num_walks} walks")
        print(f"Distinguished points collected: {len(distinguished_points)}")
    
    return None


def dlp_batch_parallel(
    alpha: int,
    beta: int,
    modulus: int,
    order: Optional[int] = None,
    max_steps_per_walk: int = 100000,
    num_walks: int = 20,
    seed: int = 0,
    verbose: bool = False
) -> Optional[int]:
    """
    Convenient wrapper for parallel variance-reduced DLP solving.
    
    Args:
        alpha: Generator
        beta: Target
        modulus: Prime modulus
        order: Group order (optional)
        max_steps_per_walk: Step budget per walk
        num_walks: Number of parallel walks
        seed: Random seed
        verbose: Print progress
        
    Returns:
        Discrete logarithm if found, None otherwise
    """
    return pollard_rho_dlp_variance_reduced(
        alpha=alpha,
        beta=beta,
        modulus=modulus,
        order=order,
        max_steps=max_steps_per_walk,
        num_walks=num_walks,
        seed=seed,
        verbose=verbose
    )


if __name__ == "__main__":
    print("Variance-Reduced Pollard's Rho for Discrete Logarithm")
    print("=" * 70)
    print()
    
    # Example 1: Small prime for demonstration
    print("Example 1: Small DLP")
    p = 1009  # Small prime
    alpha = 11  # Generator (not verified to be primitive root)
    gamma_true = 123  # Unknown to algorithm
    beta = pow(alpha, gamma_true, p)
    
    print(f"  Solving: {alpha}^γ ≡ {beta} (mod {p})")
    print(f"  True value: γ = {gamma_true}")
    
    result = dlp_batch_parallel(
        alpha=alpha,
        beta=beta,
        modulus=p,
        order=p-1,
        max_steps_per_walk=50000,
        num_walks=5,
        verbose=False
    )
    
    if result is not None:
        print(f"  ✓ Found: γ = {result}")
        # Verify
        if pow(alpha, result, p) == beta:
            print(f"  ✓ Verification passed!")
        else:
            print(f"  ✗ Verification failed!")
    else:
        print(f"  ✗ No solution found within budget")
    print()
    
    # Example 2: Slightly larger
    print("Example 2: Medium DLP")
    p = 10007  # Larger prime
    alpha = 5
    gamma_true = 4567
    beta = pow(alpha, gamma_true, p)
    
    print(f"  Solving: {alpha}^γ ≡ {beta} (mod {p})")
    print(f"  True value: γ = {gamma_true}")
    
    result = dlp_batch_parallel(
        alpha=alpha,
        beta=beta,
        modulus=p,
        order=p-1,
        max_steps_per_walk=100000,
        num_walks=10,
        verbose=False
    )
    
    if result is not None:
        print(f"  ✓ Found: γ = {result}")
        if pow(alpha, result, p) == beta:
            print(f"  ✓ Verification passed!")
        else:
            print(f"  ✗ Verification failed!")
    else:
        print(f"  ✗ No solution found within budget")
    print()
    
    print("=" * 70)
    print("Note: Expected complexity is O(√n) for group order n.")
    print("For 256-bit groups, this is ~2^128 operations.")
    print("Variance reduction improves collision probability within fixed budget.")
    print()
    print("SECURITY: This does NOT break modern ECC or RSA.")
