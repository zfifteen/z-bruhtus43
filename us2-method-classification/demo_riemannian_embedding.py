from __future__ import annotations

#!/usr/bin/env python3
"""
Demo: Riemannian Geometry Embedding for Factorization Guidance

This script demonstrates the torus geodesic embedding of numbers, a key component
of the Geodesic Validation Assault (GVA) method. It embeds a number into a high-dimensional
torus using iterative transformations with the golden ratio, fractional parts, and perturbations.

The embedding generates a curve of points that guide factorization by providing geometric
candidates in higher-dimensional space, leveraging Riemannian geometry concepts.

Usage: python us2-method-classification/demo_riemannian_embedding.py
"""

import mpmath as mp

mp.mp.dps = 100
PHI = (mp.sqrt(5) + 1) / 2
E_SQUARED = mp.e ** 2

def fractional_part(x: mp.mpf) -> mp.mpf:
    # works for mp.mpf
    return x - mp.floor(x)

def adaptive_k(n: mp.mpf, w: mp.mpf = mp.mpf("1")) -> mp.mpf:
    # log base-2 via mp.log(x, 2)
    return mp.mpf('0.3') / mp.log(mp.log(n + 1, 2), 2) * w

def embed_torus_geodesic(n: mp.mpf, k: mp.mpf, dims: int = 17) -> list:
    """
    Embed number n into a torus geodesic curve.

    Args:
        n: Number to embed (mp.mpf)
        k: Adaptive scaling factor (mp.mpf)
        dims: Dimensionality of the torus

    Returns:
        List of points on the geodesic curve
    """
    # Initialize x = n / e^2 for scaling
    x = n / E_SQUARED
    curve = []
    stabilization_count = 0
    for i in range(dims):
        frac = fractional_part(x / PHI)
        frac_pow = mp.power(frac if frac != 0 else mp.mpf(1), k)
        new_x = PHI * frac_pow
        if mp.fabs(new_x - PHI) < mp.mpf('1e-10'):
            stabilization_count += 1
            if stabilization_count > 2:
                print(f"Warning: Iteration stabilizing to PHI at i={i}, curve diversity reduced.")
        else:
            stabilization_count = 0
        x = new_x
        perturbation = mp.mpf('1e-2') * mp.sin( (k * i + n * mp.mpf('1e-15')) % (2 * mp.pi) )
        base = fractional_part(x + perturbation)
        point = []
        for j in range(dims):
            coord = fractional_part(base + j * mp.mpf('0.1'))
            point.append(coord)
        curve.append(point)
    return curve

def compute_simple_curvature(curve: list) -> list:
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
            d1 = p1[d] - p0[d]
            d2 = p2[d] - p1[d]
            curvature += abs(d2 - d1)  # Simple difference
        curvatures.append(curvature / len(p0))
    return curvatures

def test_large_n() -> None:
    """Test with a large n to check for stabilization issues.
    Uses RSA-100 semiprime: 1522605027922533360535618378132637429718068114961380688657908494580122963258952897654003790690177217898916856898458221269681967
    Factors: 37975227936943673922808872755445627854565536638199 x 40094690950920881030683735292761468389214899724061
    Source: RSA-100 challenge for reproducibility per empirical validation axiom."""
    large_n = mp.mpf('1522605027922533360535618378132637429718068114961380688657908494580122963258952897654003790690177217898916856898458221269681967')  # RSA-100 semiprime
    print(f"Testing large n: {large_n}")
    k = adaptive_k(large_n)
    print(f"Adaptive k for large n: {k}")
    curve = embed_torus_geodesic(large_n, k, dims=5)
    curvatures = compute_simple_curvature(curve)
    max_curv = max(curvatures) if curvatures else 0
    print(f"Max curvature for large n: {float(max_curv):.6f}")
    if max_curv < 1e-6:
        print("Note: Low variance—adjust perturbation amplitude for better diversity.")
    print()

def main() -> None:
    # Example number: a small semiprime for demonstration
    n = mp.mpf('271628755242544365861866007260721971360')  # Random 128-bit composite
    print(f"Embedding number: {n}")
    print(f"# Note: Large n for scaling demonstration")
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
        print(f"  Point {i}: [{', '.join(f'{float(coord):.6f}' for coord in point)}]")
    print()

    # Compute approximate curvatures
    curvatures = compute_simple_curvature(curve)
    print("Approximate curvatures along the geodesic:")
    for i, curv in enumerate(curvatures):
        print(f"  Segment {i}: {float(curv):.6f}")
    print()

    # Demonstrate guidance for factorization
    print("Factorization Guidance:")
    print("- This is a didactic geodesic/curvature sketch, not a proof of subexponential behavior.")
    print("- The embedded curve represents n in geometric space.")
    print("- Curvatures indicate 'interesting' regions for prime candidates.")
    print("- In GVA, these guide Monte Carlo sampling for factor finding.")
    print("- High curvature points may correspond to factorization breakthroughs.")

    # Test large n

    test_large_n()

if __name__ == "__main__":
    main()
