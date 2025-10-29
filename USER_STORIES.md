# User Story: Create Jargon-Free Summary of Factorization Results

## Desired Measurable Outcome

Produce a concise summary of factorization results that avoids technical jargon, ensuring 95% of the content uses everyday language readable by a general audience, with a word count under 800 for clarity.

## Underlying Reasoning

Heavy jargon in issues reduces accessibility, making it harder for others to understand and accept the work; a simplified summary promotes broader engagement and validation.

## Artifacts Created/Modified

- Created: `summary_factorization_results.md` file in the repository's docs folder.
- Modified: None.

## Data Used to Test

- Input from existing GitHub issues and results related to factorization experiments.

## Full Verifiable Output

- Input Parameters: Source issues (e.g., list of issue URLs or IDs containing raw results); target word limit (800); readability check tool (e.g., Flesch-Kincaid score aiming for grade 8 level).
- Complete Test Output: Full text of the generated summary; readability score report; third-party verification via manual review or tool output showing jargon avoidance (e.g., list of flagged terms and replacements).

---

## User Story: Classify Factorization Method as Constant Factor or Subexponential

## Desired Measurable Outcome

Determine and document whether the method is a filter on trial division (constant factor improvement, exponential time) or a new/augmented algorithm (e.g., NFS/ECM variant) with subexponential time, with classification supported by a 1-page rationale.

## Underlying Reasoning

Clear classification helps evaluate the method's novelty and interest; constant factor improvements are less compelling for large bit sizes, while subexponential ones warrant deeper investigation.

## Artifacts Created/Modified

- Created: `method_classification.md` file in the repository's analysis folder.
- Modified: None.

## Data Used to Test

- Algorithm description from existing code and issues; theoretical time complexity analysis.

## Full Verifiable Output

- Input Parameters: Algorithm pseudocode or code snippet; bit size examples (e.g., 64-bit semiprimes); time complexity formulas.
- Complete Test Output: Classification result (e.g., "Constant factor"); supporting rationale text; runtime logs for small examples showing time behavior; third-party verification via independent complexity calculation (e.g., Big-O notation proof).

---

## User Story: Generate Benchmark Dataset of 128-Bit Semiprimes

## Desired Measurable Outcome

Create a dataset of 100 semiprimes, each 128 bits, formed from two random 64-bit primes (uncorrelated, no offsets), verifiable for randomness and correctness.

## Underlying Reasoning

Standardized datasets ensure fair benchmarking; starting at 128 bits allows quick factoring to prove results before scaling, focusing on real-world semiprime challenges.

## Artifacts Created/Modified

- Created: `semiprimes_128bit_dataset.json` file in the repository's data folder, containing semiprime values and their prime factors.
- Modified: None.

## Data Used to Test

- Random prime generation library (e.g., built-in crypto tools); seed for reproducibility.

## Full Verifiable Output

- Input Parameters: Bit length (128); number of semiprimes (100); random seed (e.g., 12345 for reproducibility).
- Complete Test Output: JSON file excerpt with first 5 entries (semiprime, prime1, prime2); primality test logs (e.g., Miller-Rabin passes); randomness verification (e.g., entropy score); third-party verification via re-running generation with same seed and comparing outputs.

---

## User Story: Run Benchmarks on Own Factorization Method for 128-Bit Semiprimes

## Desired Measurable Outcome

Measure average time to factor all 100 semiprimes in the dataset, achieving a benchmark report with mean time under 1 minute per semiprime (or report actual), including variance.

## Underlying Reasoning

Solid benchmarks on standardized semiprimes validate performance claims; this is the only metric that matters for credibility, starting small to build evidence.

## Artifacts Created/Modified

- Created: `own_method_benchmarks_128bit.csv` file in the repository's benchmarks folder, with columns for semiprime, time, success.
- Modified: Factorization code if minor tweaks needed for timing.

## Data Used to Test

- Generated 128-bit semiprime dataset; hardware specs (e.g., CPU model, RAM).

## Full Verifiable Output

- Input Parameters: Dataset file path; number of runs (e.g., 3 for averaging); timeout threshold (e.g., 5 minutes per semiprime).
- Complete Test Output: CSV file contents; summary stats (mean, median, std dev time); log of each factorization run (semiprime input, factors output, elapsed time); third-party verification via script re-execution on same hardware/dataset.

---

## User Story: Set Up and Run ECM Benchmarks for 128-Bit Semiprimes

## Desired Measurable Outcome

Implement and execute ECM (Elliptic Curve Method) on the 100-semiprime dataset, producing a benchmark report with average factoring time, using a high-quality implementation in the same language as the own method.

## Underlying Reasoning

Comparison to state-of-the-art like ECM is required for novelty claims; fair conditions (same language) ensure legitimate time savings can be proven.

## Artifacts Created/Modified

- Created: `ecm_benchmarks_128bit.csv` file in the repository's benchmarks folder.
- Modified: Integration script for ECM library if needed.

## Data Used to Test

- 128-bit semiprime dataset; ECM library (e.g., GMP-ECM).

## Full Verifiable Output

- Input Parameters: Dataset file; ECM parameters (e.g., B1 bound=1e6); number of curves (e.g., 100).
- Complete Test Output: CSV with times and successes; summary stats; full run logs (input semiprime, found factors, time); third-party verification via independent ECM run with same params.

---

## User Story: Set Up and Run Cado-NFS Benchmarks for 128-Bit Semiprimes

## Desired Measurable Outcome

Execute Cado-NFS on the 100-semiprime dataset, generating a benchmark report with average time, adapted to the same programming language/environment for fair comparison.

## Underlying Reasoning

State-of-the-art comparison to Cado-NFS validates any time savings; this establishes if the method offers a constant factor or better improvement.

## Artifacts Created/Modified

- Created: `cado_nfs_benchmarks_128bit.csv` file in the repository's benchmarks folder.
- Modified: Wrapper script for Cado-NFS integration.

## Data Used to Test

- 128-bit semiprime dataset; Cado-NFS tool.

## Full Verifiable Output

- Input Parameters: Dataset file; Cado-NFS config (e.g., default for small numbers); parallelism level (e.g., 4 threads).
- Complete Test Output: CSV with results; summary stats; detailed logs per semiprime; third-party verification via re-running Cado-NFS with identical setup.

---

## User Story: Compare Benchmarks Across Methods for 128-Bit Semiprimes

## Desired Measurable Outcome

Produce a comparison report showing time differences (e.g., own method 20% faster than ECM), with statistical significance (p-value < 0.05), deciding if novelty is proven.

## Underlying Reasoning

Direct comparison against SOTA determines if there's a legitimate time save worth publishing; this cleans up claims for higher bit sizes.

## Artifacts Created/Modified

- Created: `benchmark_comparison_128bit.md` file in the repository's reports folder, including tables/charts.
- Modified: None.

## Data Used to Test

- Benchmark CSV files from own method, ECM, and Cado-NFS.

## Full Verifiable Output

- Input Parameters: CSV file paths; statistical test (e.g., t-test).
- Complete Test Output: Markdown report with tables (e.g., mean times); charts; p-values; full stats output; third-party verification via re-analysis of CSVs.

---

## User Story: Assess and Report Probabilistic Metrics if Applicable

## Desired Measurable Outcome

If method is probabilistic, calculate and document failure time T, success percentage on dataset, and comparison to truncated baseline (e.g., ECM cut off at T), showing if more/fewer semiprimes are factored.

## Underlying Reasoning

For probabilistic claims, specific metrics ensure fair evaluation against trivial baselines; this clarifies if the method offers real advantages.

## Artifacts Created/Modified

- Created: `probabilistic_metrics_128bit.md` file in the repository's analysis folder.
- Modified: None (or benchmark scripts if needed).

## Data Used to Test

- 128-bit semiprime dataset; benchmark results.

## Full Verifiable Output

- Input Parameters: Failure threshold T (e.g., average time to fail); baseline method (e.g., ECM at T).
- Complete Test Output: Report with success % (e.g., 75%); T value; comparison table (own vs. baseline factored count); logs of failures; third-party verification via re-computation with same T.
