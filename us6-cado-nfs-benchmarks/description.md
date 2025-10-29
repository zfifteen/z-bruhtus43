# User Story: Set Up and Run Cado-NFS Benchmarks for 128-Bit Semiprimes

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

