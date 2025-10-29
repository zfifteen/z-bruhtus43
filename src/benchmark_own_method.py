#!/usr/bin/env python3
"""
Script to benchmark the 'own' factorization method on the 128-bit semiprime dataset.
Uses sympy.factorint as a placeholder for the custom method.
Times each factorization and saves results to CSV.
"""

import csv
import json
import time
import sympy

def load_dataset(filepath):
    """Load the semiprime dataset from JSON."""
    with open(filepath, 'r') as f:
        return json.load(f)

def factorize_semiprime(semiprime_str):
    """
    Placeholder factorization function using sympy.factorint.
    In a real implementation, replace with the custom method.
    """
    n = int(semiprime_str)
    start_time = time.time()
    factors = sympy.factorint(n)
    end_time = time.time()
    elapsed = end_time - start_time

    # Check if factored (should have exactly two prime factors)
    if len(factors) == 2 and all(exp == 1 for exp in factors.values()):
        primes = list(factors.keys())
        success = True
    else:
        primes = []
        success = False

    return elapsed, success, primes

def main():
    dataset_path = "../data/semiprimes_128bit_dataset.json"
    output_path = "../benchmarks/own_method_benchmarks_128bit.csv"

    dataset = load_dataset(dataset_path)
    results = []

    for entry in dataset:
        semiprime = entry['semiprime']
        elapsed, success, primes = factorize_semiprime(semiprime)
        results.append({
            'semiprime': semiprime,
            'time': elapsed,
            'success': success,
            'factors': ','.join(map(str, primes)) if primes else ''
        })

    # Write to CSV
    with open(output_path, 'w', newline='') as csvfile:
        fieldnames = ['semiprime', 'time', 'success', 'factors']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)

    print(f"Benchmarks completed and saved to {output_path}")
    # Summary
    total_time = sum(r['time'] for r in results)
    avg_time = total_time / len(results)
    success_rate = sum(1 for r in results if r['success']) / len(results)
    print(f"Average time: {avg_time:.4f} seconds")
    print(f"Success rate: {success_rate:.2%}")

if __name__ == "__main__":
    main()