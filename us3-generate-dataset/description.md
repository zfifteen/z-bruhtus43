# User Story: Generate Benchmark Dataset of 128-Bit Semiprimes

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

