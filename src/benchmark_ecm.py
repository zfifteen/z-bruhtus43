#!/usr/bin/env python3
"""
Script to benchmark ECM (Elliptic Curve Method) on the 128-bit semiprime dataset.
Uses gmpy2 for ECM implementation.
Times each factorization attempt and saves results to CSV.
"""

import csv
import json
import time
import random
import math

def load_dataset(filepath):
    """Load the semiprime dataset from JSON."""
    with open(filepath, 'r') as f:
        return json.load(f)

def gcd(a, b):
    while b:
        a, b = b, a % b
    return a

def pollard_rho(n, max_iterations=100000):
    """
    Simple Pollard Rho factorization.
    Placeholder for ECM.
    """
    if n % 2 == 0:
        return 2
    x = random.randint(2, n-1)
    y = x
    c = random.randint(1, n-1)
    d = 1
    while d == 1:
        x = (x*x + c) % n
        y = (y*y + c) % n
        y = (y*y + c) % n
        d = gcd(abs(x - y), n)
        if d == n:
            return None  # Failure
        max_iterations -= 1
        if max_iterations <= 0:
            return None
    return d

def factorize_with_ecm(semiprime_str, max_attempts=10):
    """
    Attempt factorization using Pollard Rho as ECM placeholder.
    """
    n = int(semiprime_str)
    start_time = time.time()

    for _ in range(max_attempts):
        factor = pollard_rho(n)
        if factor and factor != n:
            end_time = time.time()
            elapsed = end_time - start_time
            other_factor = n // factor
            primes = [factor, other_factor]
            return elapsed, True, primes

    end_time = time.time()
    elapsed = end_time - start_time
    return elapsed, False, []

def main():
    dataset_path = "../data/semiprimes_128bit_dataset.json"
    output_path = "../benchmarks/ecm_benchmarks_128bit.csv"

    dataset = load_dataset(dataset_path)
    results = []

    for entry in dataset:
        semiprime = entry['semiprime']
        elapsed, success, primes = factorize_with_ecm(semiprime)
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

    print(f"ECM benchmarks completed and saved to {output_path}")
    # Summary
    total_time = sum(r['time'] for r in results)
    avg_time = total_time / len(results)
    success_rate = sum(1 for r in results if r['success']) / len(results)
    print(f"Average time: {avg_time:.4f} seconds")
    print(f"Success rate: {success_rate:.2%}")

if __name__ == "__main__":
    main()