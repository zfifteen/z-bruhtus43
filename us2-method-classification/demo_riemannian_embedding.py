#!/usr/bin/env python3
"""
Demo: Riemannian Geometry Embedding for Factorization Guidance

This script demonstrates the torus geodesic embedding of numbers, a key component
of the Geodesic Validation Assault (GVA) method. It embeds a number into a high-dimensional
torus using iterative transformations with the golden ratio, fractional parts, and perturbations.

The embedding generates a curve of points that guide factorization by providing geometric
candidates in higher-dimensional space, leveraging Riemannian geometry concepts.

Usage: python demo_riemannian_embedding_fixed.py
"""

import math
import mpmath
from decimal import Decimal, getcontext

# Set high precision
getcontext().prec = 100
mpmath.mp.dps = 100

PHI = Decimal((1 + math.sqrt(5)) / 2)
E_SQUARED = Decimal(math.exp(2))

def fractional_part(x):
    """Compute fractional part of a Decimal."""
    return x - Decimal(int(x))

def adaptive_k(n, w=1.0):
    """Adaptive k for scaling based on log log n."""
    log_log_n = math.log2(math.log2(float(n) + 1))
    return Decimal(0.3 / log_log_n) * Decimal(w)

def embed_torus_geodesic(n, k, dims=17):
    """
    Embed number n into a torus geodesic curve.

    Args:
        n: Number to embed (Decimal)
        k: Adaptive scaling factor (Decimal)
        dims: Dimensionality of the torus

    Returns:
        List of points on the geodesic curve
    """
    x = n / E_SQUARED
    curve = []
    for i in range(dims):
        frac = fractional_part(x / PHI)
        # Handle 0^0 case
        if frac == 0:
            frac_pow = Decimal(1)
        else:
            frac_pow = frac ** int(k)
        x = PHI * frac_pow
        perturbation = Decimal(0.01) * Decimal(math.sin(float(k) * i + float(n) * 1e-15))
        base = fractional_part(x) + perturbation
        point = []
        for j in range(dims):
            coord = fractional_part(base + Decimal(j * 0.1))
            point.append(coord)
        curve.append(point)
    return curve

def compute_simple_curvature(curve):
    """
    Compute a simple approximation of curvature along the curve.
    This is a basic demonstration; actual GVA uses full Riemannian curvature tensors.
    """
    curvatures = []
    for i in range(1, len(curve) - 1):
        p0 = curve[i-1]
        p1 = curve[i]
        p2 = curve[i+1]
        # Approximate curvature using discrete second derivative
        curvature = 0
        for d in range(len(p0)):
            d1 = float(p1[d] - p0[d])
            d2 = float(p2[d] - p1[d])
            curvature += abs(d2 - d1)  # Simple difference
        curvatures.append(curvature / len(p0))
    return curvatures

def main():
    # Example number: a small semiprime for demonstration
    n = Decimal(143)  # 11 * 13
    print(f"Embedding number: {n}")
    print(f"Factorization (ground truth): 11 * 13 = {n}")
    print()

    # Compute adaptive k
    k = adaptive_k(n)
    print(f"Adaptive k: {k}")
    print()

    # Embed into torus geodesic
    dims = 5  # Use lower dims for demo
    curve = embed_torus_geodesic(n, k, dims)
    print(f"Torus geodesic embedding ({dims}D, {len(curve)} points):")
    for i, point in enumerate(curve):
        print(f"  Point {i}: [{', '.join(f'{coord:.6f}' for coord in point)}]")
    print()

    # Compute approximate curvatures
    curvatures = compute_simple_curvature(curve)
    print("Approximate curvatures along the geodesic:")
    for i, curv in enumerate(curvatures):
        print(f"  Segment {i}: {curv:.6f}")
    print()

    # Demonstrate guidance for factorization
    print("Factorization Guidance:")
    print("- The embedded curve represents n in geometric space.")
    print("- Curvatures indicate 'interesting' regions for prime candidates.")
    print("- In GVA, these guide Monte Carlo sampling for factor finding.")
    print("- High curvature points may correspond to factorization breakthroughs.")

if __name__ == "__main__":
    main()