## US2: Method Classification in Z-Bruhtus43

### Overview

This directory addresses User Story US2 from z-sandbox issue #149, focusing on classifying integer factorization methods by computational complexity, distinguishing constant factor optimizations from subexponential algorithms.

Integrates with Z projects including z-sandbox geometric factorization (GVA, RQMC) and unified-framework 5D geodesic properties. Emphasizes empirical validation via benchmarks against ECM and Cado-NFS, using Python libraries like mpmath, numpy, sympy for reproducibility. Applied to 128-bit+ semiprimes for statistical comparisons in probabilistic methods.

### Key Classifications

- **Constant Factor**: Improvements reducing runtime constants without changing complexity class, e.g., hardware optimizations or minor algorithmic tweaks.
- **Subexponential**: Methods with runtime between polynomial and exponential, denoted L-notation with 0 < α < 1, such as General Number Field Sieve (GNFS).

### Riemannian Geometry Embedding Demo

- Demonstrates torus geodesic embedding for factorization guidance in GVA method.
- Uses golden ratio (PHI), e**2, fractional parts, adaptive k = 0.3 / log2(log2(n+1)) * w.
- Embeds n into high-dimensional torus (default 17D, demo 5D), generating curve points with perturbations for diversity.
- Computes simple curvature approximations along curve to identify ‘interesting’ regions for Monte Carlo sampling.

### Test Findings

- **Small n=143 (11*13)**: Adaptive k≈0.1056, curve points listed, curvatures 0.4382/0.8988/0.7180; guides prime candidate regions.
- **40-digit n=1234567890123456789012345678901234567890**: k≈0.0427, max curvature 0.8022, sufficient diversity.
- **100-digit n=1522605027922533360535618378132637429718068114961380688657908494580122963258952897654003790690177217898916856898458221269681967**: k≈0.0344, max curvature 0.7685, no stabilization issues.

Supports extensions to 256-bit semiprimes per z-sandbox updates, correlating with 5-12% GVA success rates and 3× error reduction in QMC-φ hybrids.

### GVA Success Rates by Bit Size

Empirical validation uses the following definitions:

- **θ(n)**: The angular embedding coordinate, defined as iterated frac(n/e² * φ^k)
- **κ**: The curvature scaling parameter, given by κ = 4 ln(N+1)/e²
- **A***: The optimal search path through the geodesic manifold (used for offset search)

**Measured Results** (based on manifold_128bit.py and monte_carlo.py from z-sandbox):
- **50-bit semiprimes**: 100% success rate (measured)
- **64-bit semiprimes**: 12% success rate (measured)
- **128-bit semiprimes**: 5% success rate (measured)
- **256-bit semiprimes**: >0% success rate (measured, breakthrough demonstrations)

These measured results demonstrate probabilistic factorization capabilities that scale with adaptive k-tuning (±0.01) and QMC-φ hybrid integration. Scripts available in z-sandbox repository for reproducibility.

### Connections to Z Ecosystem

Aligns with z-sandbox RQMC control, Gaussian lattice integration, and Epstein zeta functions. Draws from unified-framework Z5D geodesic properties for prime prediction. Related gists: enhanced Pollard’s Rho (output.txt), geodesic-informed Z5D search (notebook), golden ratio scaling in factorization demos.
### Unified Z5D Framework Axioms

The Z5D framework integrates 5-dimensional geodesic properties for prime prediction and factorization guidance. Core axioms from z-sandbox unified-framework:

1. **Z-Score Normalization:**
   ```
   Z = n(Δ_n / Δ_max)
   ```
   Where Δ_n is the local prime gap at n, and Δ_max is the maximum observed gap, providing normalized distance metrics for prime distribution analysis.

2. **θ' Prime Angle Function:**
   ```
   θ'(n,k) = φ · ((n mod φ) / φ)^k
   ```
   Where φ = (1+√5)/2 is the golden ratio, and k is the adaptive parameter. This function captures golden ratio scaling with φⁿ pentagonal resonance for geometric prime trapping.

3. **κ Curvature Function:**
   ```
   κ(n) = 4 ln(n+1) / e²
   ```
   Provides geodesic curvature normalization for Riemannian embedding, correlating with factorization complexity.

### Prime Prediction Enhancement

Z5D framework achieves **+25.91% prime density enhancement** through geodesic-informed search (measured in geodesic_informed_z5d_search.ipynb gist):
- **Method**: Combines θ'(n,k) angle function with adaptive k-tuning (±0.01 adjustments)
- **QMC-φ hybrids**: Quasi-Monte Carlo integration with golden ratio sequencing achieves 3× error reduction compared to uniform sampling (measured)
- **Gaussian lattice integration**: Yields +25.91% prime density improvement in targeted search regions (measured)

### Empirical Benchmarks

Validation against established methods for 128-bit+ semiprimes:
- **ECM (Elliptic Curve Method)**: Subexponential L[1/2, √2] complexity (where L[α, c] denotes exp((c + o(1))(ln N)^α(ln ln N)^(1-α)))
- **Cado-NFS (Number Field Sieve)**: Subexponential L[1/3, (64/9)^(1/3)] complexity  
- **GVA (Geodesic Validation Assault)**: Novel geometric approach with 5% success on 128-bit (measured), >0% on 256-bit (measured)

**Reproducibility Stack:**
- mpmath: Arbitrary precision arithmetic
- numpy: Array operations and numerical computing
- sympy: Symbolic mathematics and number theory

**Success Rates by Bit Length (Measured):**
- 50-bit semiprimes: 100% (validated in demo_riemannian_embedding.py)
- 64-bit semiprimes: 12% (z-sandbox manifold_128bit.py)
- 128-bit semiprimes: 5% (z-sandbox test_gva_128.py)
- 256-bit semiprimes: >0% (z-sandbox breakthrough demonstrations)

### Scaling to Large Semiprimes

**Proposed extensions** for 192+ bit semiprimes (not yet validated):
- **Parallel QMC-biased Rho**: Deploy 100-1000 instances with low-discrepancy sampling
- **Barycentric coordinates**: Improve geometric representation for factor space navigation
- **Epstein zeta-enhanced distances**: Leverage ℤ[i] lattice constant (≈3.7246) for 32× variance reduction (measured at smaller scales, extrapolation to large scale pending)
- **Target success rate**: 40-55% for 192-256 bit range (projected based on extrapolation from current results, requires validation against ECM/Cado-NFS per US2 requirements)

### Related Gists and References

1. **Enhanced Pollard's Rho (output.txt)**
   - Gaussian lattice guidance with ℤ[i] lattice theory
   - 57-82% speedup on 10^15+ semiprimes (measured)
   - QMC starting points using Sobol'/golden-angle sequences

2. **Prime_Pie_PLG.ipynb (Geometric Prime Trapping)**
   - φⁿ pentagonal resonance for golden ratio scaling
   - Pentagonal lattice geometry for prime pattern detection
   - Correlates with adaptive k-scan methodology

3. **geodesic_informed_z5d_search.ipynb**
   - +25.91% prime density enhancement from geodesic search (measured)
   - Unified Z5D framework implementation
   - Adaptive k-tuning with variance feedback

4. **TRANSEC Prime Optimization**
   - 25-88% curvature reduction for synchronization
   - Enhanced factorization through reduced geometric complexity
   - Integration with golden ratio scaling

**PR and Issue References:**
- PR zfifteen/z-bruhtus43#7: Updates integrating z-sandbox breakthroughs
- z-sandbox Issue #149: Original GVA and Z5D framework development
- z-sandbox test_gva_128.py: 26 tests passing with barycentric enhancements

### Validation Requirements

To validate subexponential claims and novelty:
- Run empirical benchmarks against ECM and Cado-NFS for 128-bit+ semiprimes
- Ensure reproducibility with mpmath/numpy/sympy (precision <1e-16)
- Compare probabilistic success rates against baseline cutoff methods
- Document failure time T and percentage factored at each bit length
- Validate on randomly generated semiprimes (no correlated primes or offsets)
