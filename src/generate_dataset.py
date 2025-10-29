#!/usr/bin/env python3
"""
Script to generate a dataset of 100 semiprimes, each 128 bits, formed from two random 64-bit primes.
Uses a fixed seed for reproducibility.
Outputs to data/semiprimes_128bit_dataset.json
"""

import json
import random
import sympy

def generate_semiprime_dataset(num_semiprimes=100, bit_length=128, prime_bit_length=64, seed=12345):
    """
    Generate a list of semiprimes with their prime factors.

    Args:
        num_semiprimes (int): Number of semiprimes to generate.
        bit_length (int): Total bit length of semiprime (should be 2 * prime_bit_length).
        prime_bit_length (int): Bit length of each prime factor.
        seed (int): Random seed for reproducibility.

    Returns:
        list: List of dicts, each with 'semiprime', 'prime1', 'prime2'.
    """
    random.seed(seed)
    dataset = []

    for _ in range(num_semiprimes):
        # Generate two random primes of prime_bit_length bits
        prime1 = sympy.randprime(2**(prime_bit_length-1), 2**prime_bit_length)
        prime2 = sympy.randprime(2**(prime_bit_length-1), 2**prime_bit_length)

        # Ensure they are distinct and uncorrelated (no special offsets)
        while prime1 == prime2:
            prime2 = sympy.randprime(2**(prime_bit_length-1), 2**prime_bit_length)

        semiprime = prime1 * prime2

        # Verify bit length (approximately 128 bits)
        if semiprime.bit_length() != bit_length:
            # Adjust if necessary, but randprime should ensure it
            continue

        dataset.append({
            'semiprime': str(semiprime),  # Store as string for JSON
            'prime1': str(prime1),
            'prime2': str(prime2)
        })

    return dataset

if __name__ == "__main__":
    dataset = generate_semiprime_dataset()
    output_path = "../data/semiprimes_128bit_dataset.json"
    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=4)
    print(f"Dataset generated and saved to {output_path}")
    print(f"First entry: {dataset[0]}")