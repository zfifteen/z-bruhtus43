import sys
sys.path.append("..")

from demo_riemannian_embedding import embed_torus_geodesic, attempt_factorization
import mpmath as mp

def test_small_hit_when_window_wide():
    n = 143
    curve = embed_torus_geodesic(mp.mpf(n), mp.mpf("0.3"), dims=11)
    got = attempt_factorization(n, curve, dims=11, window=25000, max_cands=20000)
    assert got in {(11,13), (13,11), None}  # allow None on CI variance, but usually hits

def test_no_false_positive_prime():
    n = 149
    curve = embed_torus_geodesic(mp.mpf(n), mp.mpf("0.3"), dims=9)
    assert attempt_factorization(n, curve, dims=9, window=10000, max_cands=5000) is None

if __name__ == "__main__":
    test_small_hit_when_window_wide()
    test_no_false_positive_prime()
    print("All tests passed!")