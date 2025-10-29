# Z-Bruhtus43: Factorization Benchmarking Project

## Overview

This repository, `z-bruhtus43`, is dedicated to addressing factorization research and benchmarking challenges outlined in the GitHub issue from the reference repository `z-sandbox`. It provides a structured approach to summarizing factorization results, classifying methods, generating datasets, and running benchmarks against state-of-the-art algorithms like ECM and Cado-NFS.

The project is inspired by and responds to the request in [z-sandbox Issue #149](https://github.com/zfifteen/z-sandbox/issues/149#issue-3564018930), emphasizing jargon-free summaries, empirical validation, and fair comparisons for novelty claims in integer factorization.

**Reference Repository:** [zfifteen/z-sandbox](https://github.com/zfifteen/z-sandbox)  
**Local Path (for reference):** /Users/velocityworks/IdeaProjects/z-sandbox

## Project Goals

- Create accessible summaries of factorization results.
- Classify factorization methods (e.g., constant factor vs. subexponential).
- Generate standardized datasets of 128-bit semiprimes.
- Benchmark custom methods against ECM and Cado-NFS.
- Compare results statistically to validate improvements.
- Handle probabilistic methods with appropriate metrics.

This aligns with core principles of empirical validation, reproducibility, and domain-specific mathematical foundations (e.g., using precise computations with tools like mpmath).

## Structure

- **PROBLEM_STATEMENT.md**: Summary of the original GitHub issue request.
- **USER_STORIES.md**: High-level user stories guiding the project.
- **Subfolders (e.g., us1-jargon-free-summary/)**: Each contains a `description.md` with detailed user story breakdowns.

## User Stories Summary

1. **US1: Jargon-Free Summary** - Produce a readable summary of results.
2. **US2: Method Classification** - Determine if the method is constant-factor or subexponential.
3. **US3: Generate Dataset** - Create 100 random 128-bit semiprimes.
4. **US4: Own Method Benchmarks** - Time factorization using the custom method.
5. **US5: ECM Benchmarks** - Run Elliptic Curve Method benchmarks.
6. **US6: Cado-NFS Benchmarks** - Run Number Field Sieve benchmarks.
7. **US7: Benchmark Comparison** - Analyze and compare results statistically.
8. **US8: Probabilistic Metrics** - Assess success rates for probabilistic methods.

## Getting Started

1. **Clone the Repo**: `git clone <repo-url>`
2. **Review Problem Statement**: Read `PROBLEM_STATEMENT.md`.
3. **Implement User Stories**: Start with dataset generation (US3) and proceed to benchmarks.
4. **Dependencies**: Python with libraries like mpmath, numpy, sympy for computations; ECM and Cado-NFS for benchmarks.

## Benchmarks and Validation

All benchmarks focus on 128-bit semiprimes from random primes. Results will be stored in CSV files under relevant subfolders. Emphasize reproducibility with seeds and precision targets (<1e-16 error).

For novelty, comparisons use fair conditions (same language, hardware). If probabilistic, report failure time (T) and success percentages against baselines.

## Contributing

Contributions welcome! Focus on empirical tests, adhering to axioms like Z = n(Δ_n / Δ_max) for discrete domains and geometric resolutions.

## License

[MIT License](LICENSE) (or specify if different).

For more details, see the reference repo: [zfifteen/z-sandbox](https://github.com/zfifteen/z-sandbox).