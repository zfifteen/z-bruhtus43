# User Story: Compare Benchmarks Across Methods for 128-Bit Semiprimes

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

