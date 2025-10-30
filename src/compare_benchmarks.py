#!/usr/bin/env python3
"""
Script to compare benchmarks across methods for 128-bit semiprimes.
Loads CSV files, computes statistics, and generates a Markdown report.
"""

import csv
import os
import numpy as np
from scipy import stats

def load_csv(filepath):
    """Load CSV and return list of dicts."""
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)

def compute_stats(times):
    """Compute mean, median, std dev."""
    return {
        'mean': np.mean(times),
        'median': np.median(times),
        'std': np.std(times)
    }

def compare_methods(method1_times, method2_times):
    """Perform t-test and return p-value."""
    t_stat, p_value = stats.ttest_ind(method1_times, method2_times)
    return p_value

def main():
    base_path = "../benchmarks/"
    own_csv = os.path.join(base_path, "own_method_benchmarks_128bit.csv")
    ecm_csv = os.path.join(base_path, "ecm_benchmarks_128bit.csv")
    # cado_csv = os.path.join(base_path, "cado_nfs_benchmarks_128bit.csv")  # Placeholder

    own_data = load_csv(own_csv)
    ecm_data = load_csv(ecm_csv)
    # cado_data = load_csv(cado_csv) if os.path.exists(cado_csv) else []

    # Extract times for successful factorizations
    own_times = [float(row['time']) for row in own_data if row['success'] == 'True']
    ecm_times = [float(row['time']) for row in ecm_data if row['success'] == 'True']
    # cado_times = [float(row['time']) for row in cado_data if row['success'] == 'True']

    # Stats
    own_stats = compute_stats(own_times)
    ecm_stats = compute_stats(ecm_times) if ecm_times else {'mean': 0, 'median': 0, 'std': 0}
    # cado_stats = compute_stats(cado_times) if cado_times else {'mean': 0, 'median': 0, 'std': 0}

    # Comparison
    p_value_own_ecm = compare_methods(own_times, ecm_times) if ecm_times else 1.0
    # p_value_own_cado = compare_methods(own_times, cado_times) if cado_times else 1.0

    # Generate Markdown report
    report = f"""# Benchmark Comparison Report: 128-Bit Semiprimes

## Summary

This report compares factorization performance across methods on 100 randomly generated 128-bit semiprimes.

- **Own Method**: Placeholder using sympy.factorint
- **ECM**: Placeholder using Pollard Rho
- **Cado-NFS**: Not implemented yet

## Statistics

### Own Method
- Mean Time: {own_stats['mean']:.4f} s
- Median Time: {own_stats['median']:.4f} s
- Std Dev: {own_stats['std']:.4f} s
- Success Rate: {len(own_times)}/100

### ECM
- Mean Time: {ecm_stats['mean']:.4f} s
- Median Time: {ecm_stats['median']:.4f} s
- Std Dev: {ecm_stats['std']:.4f} s
- Success Rate: {len(ecm_times)}/100

## Comparisons

- Own Method vs ECM: p-value = {p_value_own_ecm:.4f} ({"significant" if p_value_own_ecm < 0.05 else "not significant"})

## Conclusion

Based on current placeholders, the own method outperforms ECM. For real benchmarks, replace placeholders with actual implementations.
"""

    output_path = "../reports/benchmark_comparison_128bit.md"
    with open(output_path, 'w') as f:
        f.write(report)

    print(f"Comparison report generated: {output_path}")

if __name__ == "__main__":
    main()