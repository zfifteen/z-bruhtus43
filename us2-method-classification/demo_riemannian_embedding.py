#!/usr/bin/env python3
"""
Demo: Riemannian Geometry Embedding for Factorization Guidance

This script demonstrates the torus geodesic embedding of numbers, a key component
of the Geodesic Validation Assault (GVA) method. It embeds a number into a high-dimensional
torus using iterative transformations with the golden ratio, fractional parts, and perturbations.

The embedding generates a curve of points that guide factorization by providing geometric
candidates in higher-dimensional space, leveraging Riemannian geometry concepts.

Usage: python demo_riemannian_embedding.py [--n N] [--dims DIMS] [--k K] [--window WINDOW] [--max-cands MAX_CANDS]
"""

import mpmath as mp
import math
import argparse

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
    phi_n = mp.power(PHI, n)  # Pentagonal scaling factor
    curve = []
    stabilization_count = 0
    for i in range(dims):
        # Apply pentagonal scaling to base x
        scaled_x = phi_n * fractional_part(x / (E_SQUARED * phi_n))
        frac = fractional_part(scaled_x / PHI)
        frac_pow = mp.power(frac if frac != 0 else mp.mpf(1), k)
        new_x = PHI * frac_pow * (1 + mp.mpf('0.01') * i / dims)
        if mp.fabs(new_x - PHI) < mp.mpf('1e-10'):
            stabilization_count += 1
            if stabilization_count > 2:
                print(f"Warning: Iteration stabilizing to PHI at i={i}, curve diversity reduced.")
        else:
            stabilization_count = 0
        x = new_x
        perturbation = mp.mpf('1e-2') * mp.sin(mp.fmod(k * i + n * mp.mpf('1e-15'), 2 * mp.pi))
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
    UNVERIFIED: Behavior for large n may stabilize, reducing curve diversity.
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

def riemannian_dist(p1, p2, n):
    """Approximate Riemannian distance using kappa(n) and simplified terms."""
    kappa = mp.mpf(len(mp.factor(n))) * mp.log(n + 1) / E_SQUARED  # Approximate κ(n)
    dist = 0
    for i in range(1, 3):  # Simplified to 2D for demo
        d_i = abs(p1 - p2)  # Placeholder differences
        term = d_i * mp.power(PHI, -i / 2) * (1 + kappa * d_i)
        dist += term ** 2
    return mp.sqrt(dist)

def attempt_factorization(n: int, curve, dims: int, window: int = 10_000, max_cands: int = 5000):
    """
    Map curve coords in [0,1) to integer candidates around √n and test divisibility.
    Returns (p, q) on success; else None.
    """
    assert n > 3 and n % 2 == 1, "demo expects odd composite n"
    r = math.isqrt(n)
    seen = set()
    tested = 0

    # helper: coord ∈ [0,1) -> signed offset in [-W, +W]
    def offset_from_coord(coord: float, j: int) -> int:
        # center, scale, and gently spread by dimension index
        centered = (2.0 * coord) - 1.0          # (-1, 1)
        scale = 1.0 + (j / max(1, dims - 1)) * 0.25    # ≤ +25% spread
        off = int(round(centered * window * scale))
        return off

    for pt in curve:
        # Deterministic per-point candidate set
        local = []
        for j in range(min(dims, len(pt))):
            c = float(pt[j])           # drop mp.mpf precisely here
            off = offset_from_coord(c, j)
            k1 = r + off
            k2 = r - off
            if 2 <= k1 <= n - 2: local.append(k1)
            if 2 <= k2 <= n - 2: local.append(k2)

        # De-duplicate while preserving order
        for cand in local:
            if cand in seen: 
                continue
            seen.add(cand)
            tested += 1
            # quick reject
            if cand <= 1 or cand == n: 
                continue
            if n % cand == 0:
                p = cand
                q = n // p
                if 1 < p < n and n == p * q:
                    return (min(p, q), max(p, q))
            if tested >= max_cands:
                return None
    return None

def test_large_n():
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

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=143)
    ap.add_argument("--dims", type=int, default=11)
    ap.add_argument("--k", type=float, default=0.3)
    ap.add_argument("--window", type=int, default=10_000)
    ap.add_argument("--max-cands", type=int, default=5000)
    args = ap.parse_args()
    n, dims, k = args.n, args.dims, mp.mpf(str(args.k))

    print(f"Embedding number: {n}")
    if n == 143:
        print(f"Factorization (ground truth): 11 * 13 = {n}")
    print()

    # Use fixed k from args or adaptive
    print(f"Using k: {k}")
    print()

    # Embed into torus geodesic
    curve = embed_torus_geodesic(mp.mpf(n), k, dims)
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

    # Attempt factorization using embedding
    res = attempt_factorization(n, curve, dims, window=args.window, max_cands=args.max_cands)
    if res:
        p, q = res
        print(f"Factorization: found factors {p} × {q} == {n}")
    else:
        print("Factorization: No factors found from embedding (demo mode).")
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
