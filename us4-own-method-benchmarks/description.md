# User Story: Run Benchmarks on Own Factorization Method for 128-Bit Semiprimes

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

