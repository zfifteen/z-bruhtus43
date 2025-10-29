# User Story: Set Up and Run ECM Benchmarks for 128-Bit Semiprimes

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

