# User Story: Assess and Report Probabilistic Metrics if Applicable

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

