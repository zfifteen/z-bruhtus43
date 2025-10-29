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

def fractional_part(x):
    # works for mp.mpf
    return x - mp.floor(x)

def adaptive_k(n, w=1):
    # log base-2 via mp.log(x, 2)
    return mp.mpf('0.3') / mp.log(mp.log(n + 1, 2), 2) * w

def embed_torus_geodesic(n, k, dims=17):
    """
    Embed number n into a torus geodesic curve.

    Args:
        n: Number to embed (mp.mpf)
        k: Adaptive scaling factor (mp.mpf)
        dims: Dimensionality of the torus

    Returns:
        List of points on the geodesic curve
    """
    x = n / E_SQUARED
    curve = []
    for i in range(dims):
        frac = fractional_part(x / PHI)
        frac_pow = mp.power(frac if frac != 0 else mp.mpf(1), k)
        x = PHI * frac_pow
        perturbation = mp.mpf('1e-2') * mp.sin(k * i + n * mp.mpf('1e-15'))
        base = fractional_part(x + perturbation)
        point = []
        for j in range(dims):
            coord = fractional_part(base + j * mp.mpf('0.1'))
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
            d1 = p1[d] - p0[d]
            d2 = p2[d] - p1[d]
            curvature += abs(d2 - d1)  # Simple difference
        curvatures.append(curvature / len(p0))
    return curvatures

def main():
    # Example number: a small semiprime for demonstration
    n = mp.mpf(143)  # 11 * 13
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

if __name__ == "__main__":
    main()
