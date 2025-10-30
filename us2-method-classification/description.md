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

### Connections to Z Ecosystem

Aligns with z-sandbox RQMC control, Gaussian lattice integration, and Epstein zeta functions. Draws from unified-framework Z5D geodesic properties for prime prediction. Related gists: enhanced Pollard’s Rho (output.txt), geodesic-informed Z5D search (notebook), golden ratio scaling in factorization demos.