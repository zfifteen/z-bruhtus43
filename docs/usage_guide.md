# Usage Guide: Variance-Reduced Pollard's Rho

## Quick Start

### Integer Factorization

```python
from variance_reduced_rho import pollard_rho_batch

# Factor a semiprime
n = 1077739877  # 32771 × 32887
result = pollard_rho_batch(
    n=n,
    max_iterations_per_walk=100000,
    num_walks=10,
    seed=42,
    verbose=True
)

if result:
    p, q = result
    print(f"Factors: {p} × {q}")
    assert p * q == n
else:
    print("No factors found within budget")
```

### Discrete Logarithm

```python
from variance_reduced_dlp import dlp_batch_parallel

# Solve α^γ ≡ β (mod p)
p = 10007  # Prime modulus
alpha = 5  # Base
gamma_true = 4567  # Unknown to algorithm
beta = pow(alpha, gamma_true, p)

result = dlp_batch_parallel(
    alpha=alpha,
    beta=beta,
    modulus=p,
    order=p-1,
    max_steps_per_walk=100000,
    num_walks=10,
    seed=42,
    verbose=True
)

if result is not None:
    print(f"Discrete log: γ = {result}")
    assert pow(alpha, result, p) == beta
else:
    print("No solution found within budget")
```

## Running Benchmarks

### Full Benchmark Suite

```bash
cd src
python3 benchmark_variance_reduced.py
```

This runs comprehensive benchmarks on:
- 40-bit, 50-bit, 60-bit semiprimes (factorization)
- 16-bit, 20-bit prime moduli (DLP)

Results are saved to `benchmarks/variance_reduced_results.json`

### Custom Benchmark

```python
from benchmark_variance_reduced import benchmark_factorization_success_rate

results = benchmark_factorization_success_rate(
    bit_size=48,
    num_trials=100,
    max_iterations=200000,
    num_walks=10
)

print(f"Success rate: {results['success_rate_percent']}%")
```

## API Reference

### variance_reduced_rho.py

#### pollard_rho_variance_reduced()

Main factorization function.

**Parameters**:
- `n` (int): The integer to factor (should be composite)
- `max_iterations` (int): Maximum iterations per walk (default: 1,000,000)
- `num_walks` (int): Number of parallel walks to attempt (default: 1)
- `use_lattice_guide` (bool): Whether to use Gaussian lattice guidance (default: True)
- `seed` (int): Random seed for reproducibility (default: 0)

**Returns**:
- `int | None`: A nontrivial factor of n, or None if no factor found

**Example**:
```python
factor = pollard_rho_variance_reduced(
    n=899,
    max_iterations=50000,
    num_walks=5,
    use_lattice_guide=True,
    seed=123
)
```

#### pollard_rho_batch()

Convenience wrapper that returns both factors.

**Parameters**:
- `n` (int): The semiprime to factor
- `max_iterations_per_walk` (int): Iteration budget per walk (default: 100,000)
- `num_walks` (int): Number of independent walks (default: 10)
- `seed` (int): Random seed (default: 0)
- `verbose` (bool): Print progress (default: False)

**Returns**:
- `tuple[int, int] | None`: Tuple of (factor1, factor2) if successful, None otherwise

**Example**:
```python
result = pollard_rho_batch(n=143, num_walks=5, verbose=True)
if result:
    p, q = result
    print(f"{n} = {p} × {q}")
```

#### SobolSequence Class

Quasi-random sequence generator.

**Methods**:
- `__init__(dimension=2, seed=0)`: Initialize generator
- `next() -> tuple[float, ...]`: Get next point in [0,1)^d
- `skip_to(index)`: Jump to specific index in sequence

**Example**:
```python
sobol = SobolSequence(dimension=2, seed=42)
for i in range(5):
    u, v = sobol.next()
    print(f"Point {i}: ({u:.6f}, {v:.6f})")
```

#### GaussianLatticeGuide Class

Lattice-guided parameter selection.

**Methods**:
- `__init__(n)`: Initialize for semiprime n
- `get_biased_constant(base_c) -> int`: Apply lattice bias to constant
- `get_geodesic_start(sobol_point) -> int`: Generate geodesic starting point

**Constants**:
- `EPSTEIN_ZETA`: Epstein zeta constant ≈ 3.7246
- `PHI`: Golden ratio φ ≈ 1.618

### variance_reduced_dlp.py

#### pollard_rho_dlp_variance_reduced()

Main DLP solver function.

**Parameters**:
- `alpha` (int): Generator (base)
- `beta` (int): Target element
- `modulus` (int): Prime modulus defining cyclic group
- `order` (int | None): Group order (default: modulus-1)
- `max_steps` (int): Maximum steps per walk (default: 1,000,000)
- `num_walks` (int): Number of parallel walks (default: 10)
- `distinguish_bits` (int): Distinguished point criterion (default: 16)
- `seed` (int): Random seed (default: 0)
- `verbose` (bool): Print progress (default: False)

**Returns**:
- `int | None`: The discrete logarithm γ such that α^γ ≡ β (mod modulus), or None

**Example**:
```python
gamma = pollard_rho_dlp_variance_reduced(
    alpha=11,
    beta=510,
    modulus=1009,
    max_steps=100000,
    num_walks=10,
    seed=42
)
```

#### dlp_batch_parallel()

Convenience wrapper for DLP solving.

**Parameters**: Same as `pollard_rho_dlp_variance_reduced()` minus `distinguish_bits`

**Returns**: Same as `pollard_rho_dlp_variance_reduced()`

#### WalkState Class

Dataclass representing walk state.

**Fields**:
- `element` (int): Current group element
- `alpha_power` (int): Exponent of α
- `beta_power` (int): Exponent of β
- `steps` (int): Number of steps taken

## Performance Tuning

### Iteration Budget

The iteration budget (`max_iterations` or `max_steps`) should be proportional to √p:

| Bit Size | Suggested Budget |
|----------|------------------|
| 32-bit   | 10,000 - 50,000  |
| 40-bit   | 50,000 - 100,000 |
| 48-bit   | 100,000 - 250,000|
| 64-bit   | 250,000 - 1M     |
| 128-bit  | 1M - 10M         |

### Number of Walks

More walks increase success probability but with diminishing returns:

- **5 walks**: Good for quick tests
- **10 walks**: Balanced default
- **20 walks**: For higher success rates
- **50+ walks**: May be needed for larger semiprimes

### Memory Considerations

**Factorization**: Memory usage is minimal, dominated by iteration state.

**DLP**: Memory scales with distinguished points collected:
- More `distinguish_bits` → fewer points → less memory
- Typical: 16-20 bits works well for medium problems
- For very large groups: use 24+ bits

### Parallelization

The `num_walks` parameter provides implicit parallelization:
- Each walk uses different Sobol initialization
- Walks are independent and could be parallelized across cores
- Current implementation is serial but could be extended to multiprocessing

## Troubleshooting

### "No factor found within budget"

**Solutions**:
1. Increase `max_iterations`
2. Increase `num_walks`
3. Try different `seed` values
4. Verify n is actually composite: `sympy.isprime(n)`

### "DLP: No solution found"

**Solutions**:
1. Increase `max_steps` and/or `num_walks`
2. Decrease `distinguish_bits` to collect more collision points
3. Verify problem parameters (α, β, modulus)
4. For large moduli, expect low success rate within reasonable budgets

### Performance is slow

**Check**:
1. Are you using appropriate bit sizes? Very large semiprimes need huge budgets
2. Is `use_lattice_guide=True`? This should help, not hurt
3. Consider using specialized algorithms (ECM, NFS) for >256-bit numbers

## Examples Gallery

### Example 1: Small Educational Semiprimes

```python
from variance_reduced_rho import pollard_rho_batch

test_cases = [
    (143, "11 × 13"),
    (899, "29 × 31"),
    (1003, "17 × 59"),
]

for n, description in test_cases:
    print(f"Factoring {n} ({description})")
    result = pollard_rho_batch(n, num_walks=3, verbose=False)
    if result:
        p, q = result
        print(f"  ✓ {p} × {q}\n")
```

### Example 2: 30-bit Semiprime

```python
# Known factorization: 1077739877 = 32771 × 32887
n = 1077739877

import time
start = time.time()
result = pollard_rho_batch(
    n=n,
    max_iterations_per_walk=100000,
    num_walks=10,
    seed=42
)
elapsed = time.time() - start

if result:
    p, q = result
    print(f"Factored {n} in {elapsed:.3f}s")
    print(f"  {p} × {q}")
```

### Example 3: Batch Processing

```python
from sympy import randprime

# Generate and factor multiple semiprimes
results = []
for trial in range(10):
    p = randprime(2**31, 2**32)
    q = randprime(2**31, 2**32)
    n = p * q
    
    result = pollard_rho_batch(n, num_walks=5, seed=trial)
    success = result is not None
    results.append(success)

success_rate = sum(results) / len(results) * 100
print(f"Success rate: {success_rate:.1f}%")
```

### Example 4: DLP in Small Group

```python
from variance_reduced_dlp import dlp_batch_parallel

# Setup
p = 1009
alpha = 11
secret = 456
beta = pow(alpha, secret, p)

print(f"Solving: {alpha}^γ ≡ {beta} (mod {p})")

result = dlp_batch_parallel(
    alpha=alpha,
    beta=beta,
    modulus=p,
    max_steps_per_walk=50000,
    num_walks=10,
    verbose=False
)

if result is not None:
    print(f"Found: γ = {result}")
    print(f"Verify: {alpha}^{result} ≡ {pow(alpha, result, p)} ≡ {beta} (mod {p})")
else:
    print("Not found within budget")
```

## Integration with Existing Code

### Using with SymPy

```python
from variance_reduced_rho import pollard_rho_batch
import sympy

n = 123456789012345
result = pollard_rho_batch(n, num_walks=10)

if result:
    p, q = result
    # Verify with sympy
    factors = sympy.factorint(n)
    print(f"Our result: {p} × {q}")
    print(f"SymPy: {factors}")
```

### Comparison with Standard Pollard's Rho

```python
import sympy
import time

n = 1077739877

# Variance-reduced version
start = time.time()
result_vr = pollard_rho_batch(n, num_walks=5)
time_vr = time.time() - start

# SymPy's implementation
start = time.time()
result_sp = sympy.factorint(n)
time_sp = time.time() - start

print(f"Variance-reduced: {time_vr:.4f}s")
print(f"SymPy: {time_sp:.4f}s")
```

## Next Steps

- Read [variance_reduction_theory.md](variance_reduction_theory.md) for theoretical background
- Run benchmarks to see empirical performance
- Experiment with different parameter settings
- Integrate into your own factorization/DLP workflows

## Support and Contributing

This implementation is part of the z-bruhtus43 research project. For questions or contributions, see the main repository README.
