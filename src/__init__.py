"""
Variance-reduced Pollard's Rho implementations.

This package provides variance-reduced algorithms for:
- Integer factorization (Pollard's Rho)
- Discrete logarithm problems (DLP)

Key features:
- RQMC seeding with Sobol/Owen sequences
- Gaussian lattice guidance
- Geodesic walk bias
"""

from .variance_reduced_rho import (
    SobolSequence,
    GaussianLatticeGuide,
    pollard_rho_variance_reduced,
    pollard_rho_batch,
    gcd
)

from .variance_reduced_dlp import (
    pollard_rho_dlp_variance_reduced,
    dlp_batch_parallel,
    WalkState,
    is_distinguished,
    partition_function,
    mod_inverse
)

__all__ = [
    # Factorization
    'SobolSequence',
    'GaussianLatticeGuide',
    'pollard_rho_variance_reduced',
    'pollard_rho_batch',
    'gcd',
    # DLP
    'pollard_rho_dlp_variance_reduced',
    'dlp_batch_parallel',
    'WalkState',
    'is_distinguished',
    'partition_function',
    'mod_inverse',
]
