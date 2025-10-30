# Issue #16 Implementation Plan: Coherence-Enhanced Pollard's Rho

## Overview

This document outlines the implementation plan for GitHub issue #16: "Enhancement of Pollard's Rho Algorithm Using Geometric Lattice and Randomized Quasi-Monte Carlo Techniques"

**Goal**: Integrate optical physics-inspired coherence control with RQMC techniques to achieve adaptive variance reduction in Pollard's Rho factorization.

## Core Objectives

1. **Coherence Parameter α**: Implement α ∈ [0,1] control for sample correlation
2. **RQMC with Owen Scrambling**: Port scrambled Sobol'/Halton sequences
3. **Adaptive Variance Control**: Maintain ~10% variance target dynamically
4. **Split-Step Evolution**: Periodic re-scrambling for stability
5. **Performance Validation**: Demonstrate 32x efficiency and O(N^-3/2) convergence

## Mathematical Framework

### Coherence-to-Scrambling Mapping

```
α ∈ [0, 1]  →  Scrambling parameters

Scrambling depth:     d(α) = ⌈32 × (1 - α²)⌉
Ensemble size:        M(α) = max(1, ⌈10 × (1 - α²)⌉)
Coherence length:     l_c ~ 1/α
Decoherence rate:     γ ~ (1 - α)
```

### Convergence Rates

- **Standard MC**: O(N^(-1/2))
- **Unscrambled QMC**: O(N^(-1) (log N)^(s-1))
- **Scrambled RQMC**: O(N^(-3/2+ε)) for smooth integrands

### Optical Physics Analogy

| Optics Concept | Factorization Application |
|----------------|--------------------------|
| Source coherence | Sample correlation strength |
| Temporal spreading | Variance amplification |
| Nonlinear dispersion | High-dimensional geometry |
| Complex screen method | Ensemble averaging |
| Split-step Fourier | Iterative refinement |

## Implementation Architecture

### Phase 1: Core Module Porting

#### 1.1 RQMC Control Module
**File**: `src/rqmc_control.py`

**Components**:
- `RQMCScrambler` - Base scrambling class with α mapping
- `ScrambledSobolSampler` - Sobol' with Owen scrambling
- `ScrambledHaltonSampler` - Halton with scrambling
- `AdaptiveRQMCSampler` - Dynamic α scheduling for 10% variance
- `SplitStepRQMC` - Periodic re-scrambling evolution

**Key Features**:
- Nested random digit scrambling (Owen method)
- Hash-based scrambling for parallel generation
- Per-dimension α control (weighted discrepancy)
- Variance estimation from replications

#### 1.2 Reduced Coherence Module
**File**: `src/reduced_coherence.py`

**Components**:
- `ReducedCoherenceSampler` - Main coherence control
- `CoherenceMetrics` - Performance tracking
- `compare_coherence_modes()` - Benchmarking utility

**Key Features**:
- Ensemble averaging (complex screen analog)
- Split-step evolution with decoherence
- Adaptive coherence based on variance feedback
- Multiple coherence modes (fully/partially/reduced/incoherent)

### Phase 2: Integration with Existing Implementations

#### 2.1 Enhanced Variance-Reduced Rho
**File**: `src/variance_reduced_rho_coherence.py`

**New Class**: `CoherenceEnhancedPollardRho`

```python
class CoherenceEnhancedPollardRho:
    def __init__(self, alpha: float = 0.5, target_variance: float = 0.1):
        """Initialize with coherence control."""
        self.alpha = alpha
        self.rqmc_sampler = ScrambledSobolSampler(alpha=alpha)
        self.coherence_sampler = ReducedCoherenceSampler(coherence_alpha=alpha)
        self.gaussian_lattice = GaussianLatticeGuide()

    def pollard_rho_adaptive_coherence(self, n: int, max_iterations: int):
        """Pollard's Rho with adaptive α control."""
        # Implementation with split-step evolution
        pass
```

**Features**:
- Combines existing RQMC seeding with α control
- Adaptive variance targeting (~10%)
- Split-step evolution between refinement stages
- Ensemble averaging for stability

#### 2.2 Unified Factorization Interface
**File**: `src/unified_coherence_factorizer.py`

**API Design**:
```python
def factor_with_coherence(
    n: int,
    alpha: float = 0.5,
    mode: str = "adaptive",
    target_variance: float = 0.1,
    num_walks: int = 10
) -> Optional[int]:
    """
    Unified interface for coherence-enhanced factorization.

    Args:
        n: Number to factor
        alpha: Coherence parameter (0=incoherent, 1=coherent)
        mode: "adaptive", "fixed", "split_step"
        target_variance: For adaptive mode (~10%)
        num_walks: Parallel walks
    """
```

### Phase 3: Testing & Validation

#### 3.1 Unit Tests
**File**: `tests/test_coherence_enhanced.py`

Test coverage:
- α-to-scrambling parameter mapping
- Owen scrambling correctness
- Ensemble averaging
- Split-step evolution
- Adaptive α scheduling
- Integration with existing factorization

#### 3.2 Benchmark Suite
**File**: `src/benchmark_coherence_performance.py`

Metrics:
- Variance reduction vs α
- Convergence rate empirical measurement
- Success rates across bit sizes (30-256 bit)
- Comparison: MC vs QMC vs RQMC (α=0.5) vs Adaptive
- 32x efficiency validation

#### 3.3 Performance Targets

| Semiprime Size | Success Rate Goal | Variance Reduction |
|----------------|------------------|-------------------|
| 30-40 bit | 100% | 40-60% |
| 50-60 bit | 80-100% | 30-50% |
| 64-128 bit | 10-20% | 20-40% |
| 256-bit | >0% | 10-30% |

### Phase 4: Documentation

#### 4.1 Technical Documentation
**File**: `docs/coherence_enhanced_pollard_rho.md`

Sections:
1. Mathematical foundations
2. Optical physics motivation
3. Implementation details
4. API reference
5. Performance analysis
6. Usage examples

#### 4.2 Integration Guide
**File**: `docs/coherence_integration_guide.md`

Content:
- How to port to other algorithms
- Parameter tuning guidelines
- When to use each coherence mode
- Troubleshooting common issues

#### 4.3 Benchmark Report
**File**: `docs/coherence_benchmark_results.md`

Data:
- Empirical convergence rates
- Variance reduction measurements
- Success rate tables
- Comparison with baseline implementations

### Phase 5: Demonstration

#### 5.1 Demo Script
**File**: `demos/coherence_enhanced_demo.py`

Demonstrations:
1. α parameter effect on variance
2. Adaptive vs fixed coherence comparison
3. Split-step evolution visualization
4. Ensemble averaging benefits
5. Full factorization workflow
6. Performance scaling

## Implementation Timeline

### Sprint 1: Core Porting (Current)
- [x] Review existing implementations
- [ ] Port `rqmc_control.py`
- [ ] Port `reduced_coherence.py`
- [ ] Basic integration tests

### Sprint 2: Integration
- [ ] Create `CoherenceEnhancedPollardRho` class
- [ ] Integrate with existing `variance_reduced_rho.py`
- [ ] Implement split-step evolution
- [ ] Unified interface

### Sprint 3: Validation
- [ ] Comprehensive test suite
- [ ] Benchmark suite
- [ ] Performance validation
- [ ] Parameter tuning

### Sprint 4: Documentation & Polish
- [ ] Technical documentation
- [ ] Usage guides
- [ ] Demo scripts
- [ ] Final benchmarks

## Success Criteria

### Must Have
✅ α parameter controls scrambling depth as specified
✅ Adaptive mode maintains ~10% variance
✅ Split-step evolution implemented and tested
✅ Variance reduction demonstrated (>30%)
✅ All tests passing (>95% coverage)

### Should Have
✅ O(N^-3/2) convergence demonstrated empirically
✅ 32x efficiency validated on benchmark suite
✅ Success rate improvements documented
✅ Comprehensive documentation

### Nice to Have
- Visualization tools for α effect
- Interactive parameter tuning UI
- Integration with other factorization methods
- Extended to DLP (discrete logarithm problem)

## Technical Dependencies

### Required Modules
- `numpy` - Array operations
- `mpmath` - High-precision arithmetic
- Existing `variance_reduced_rho.py`
- Existing `low_discrepancy.py` (Sobol', golden-angle)

### Optional Modules
- `scipy` - Statistical tests
- `matplotlib` - Visualization (for demos)
- `sympy` - Prime generation (for benchmarks)

## Risk Assessment

### Low Risk
- RQMC control module: Already implemented and tested in z-sandbox
- Reduced coherence module: Proven in z-sandbox
- Integration points: Clear APIs exist

### Medium Risk
- Performance validation: May need parameter tuning
- Convergence rate measurement: Requires careful statistics
- 32x efficiency claim: May be benchmark-specific

### Mitigation Strategies
1. Port proven implementations first (reduce implementation risk)
2. Start with small semiprimes for validation
3. Extensive unit testing before benchmarking
4. Document all assumptions and limitations clearly
5. Use reproducible seeds throughout

## References

### Issue & Requirements
- GitHub Issue #16: Coherence parameter integration
- z-sandbox implementations (proven baseline)
- arXiv:2503.02629 (optical physics theory)

### Mathematical Foundations
- Owen (1997): Scrambled nets
- Joe & Kuo (2008): Sobol' sequences
- Dick (2010): RQMC convergence analysis
- Epstein zeta functions for lattice theory

### Code References
- z-sandbox: `python/rqmc_control.py` (775 lines)
- z-sandbox: `python/reduced_coherence.py` (498 lines)
- z-sandbox: `python/pollard_gaussian_monte_carlo.py` (500+ lines)
- z-bruhtus43: `src/variance_reduced_rho.py` (338 lines)

---

**Status**: Phase 1 in progress
**Last Updated**: 2025-10-30
**Next Steps**: Port RQMC control module to z-bruhtus43
