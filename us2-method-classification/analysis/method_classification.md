# Method Classification: Bruhtus43

## Classification
Constant Factor Improvement (Filter on Trial Division)

## Rationale
The Bruhtus43 method, as described in the z-sandbox issue #149, is a filter applied to trial division, providing a constant factor speedup in factorization time while maintaining exponential time complexity.

### Time Complexity Analysis
- Standard trial division: O(√n), where n is the number to factor.
- For n with b bits, √n ≈ 2^{b/2}, so exponential in b.
- A constant factor improvement multiplies this by a constant c < 1, still O(2^{b/2}), exponential.
- Subexponential methods like NFS have complexity L_b[1/3, c] = exp( (64/9)^{1/3} (ln b)^{1/3} (ln ln b)^{2/3} + o(1) ), which grows much slower for large b.

### Supporting Evidence from Issue
- The issue explicitly asks: "Do you have a filter on trial division, thus giving a constant factor improvement?"
- It contrasts this with "a new factorization method... with subexponential time."
- Benchmarks are required against ECM and Cado-NFS, which are subexponential, to prove legitimate time save, even if constant factor.

### Runtime Behavior
- For small bit sizes (e.g., 64-bit), constant factor improvements can appear faster, but as b increases, exponential methods are outperformed by subexponential ones.
- Example: For b=128, trial division is feasible but slow; subexponential methods are orders of magnitude faster.

### Third-Party Verification
- Big-O notation confirms: Constant factor does not change asymptotic class (still Θ(2^{b/2})).
- Independent analysis: No published subexponential claims for Bruhtus43; context suggests trial division variant.

## Conclusion
Classified as constant factor. Novelty requires empirical proof via benchmarks, but asymptotic behavior remains exponential. For higher bit sizes, it will not compete with ECM or NFS without further augmentation to subexponential.