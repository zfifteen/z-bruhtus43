#!/usr/bin/env python3
"""
QMC vs Monte Carlo Visualization Demo

This script provides a conceptual demonstration of how Quasi-Monte Carlo (QMC)
low-discrepancy sequences provide better coverage of parameter space compared
to pseudo-random Monte Carlo sampling.

The visualization shows how Halton sequences fill 2D parameter space more uniformly
than random sampling, which directly translates to better exploration in Pollard's Rho
parameter selection (c, x0).

Usage: python src/qmc_visualization_demo.py
"""

import numpy as np

try:
    from scipy.stats import qmc
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


def halton_sequence_fallback(index: int, base: int) -> float:
    """Generate Halton sequence value for given index and base."""
    result = 0.0
    f = 1.0 / base
    i = index
    while i > 0:
        result += f * (i % base)
        i //= base
        f /= base
    return result


def generate_monte_carlo_points(n_points: int, dimensions: int = 2, seed: int = 42) -> np.ndarray:
    """Generate pseudo-random Monte Carlo points."""
    np.random.seed(seed)
    return np.random.random((n_points, dimensions))


def generate_halton_points(n_points: int, dimensions: int = 2) -> np.ndarray:
    """Generate Halton sequence points."""
    if HAS_SCIPY:
        sampler = qmc.Halton(d=dimensions, scramble=False, seed=42)
        return sampler.random(n_points)
    else:
        # Fallback implementation
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        points = np.zeros((n_points, dimensions))
        for i in range(n_points):
            for d in range(dimensions):
                points[i, d] = halton_sequence_fallback(i + 1, primes[d])
        return points


def compute_discrepancy(points: np.ndarray, n_bins: int = 10) -> float:
    """
    Compute a simple measure of discrepancy (non-uniformity).
    Lower values indicate more uniform coverage.
    
    This is a simplified star-discrepancy approximation using binning.
    """
    n_points = len(points)
    dimensions = points.shape[1]
    
    # Create bins
    bin_counts = np.zeros([n_bins] * dimensions)
    
    # Count points in each bin
    for point in points:
        bin_idx = tuple(min(int(p * n_bins), n_bins - 1) for p in point)
        bin_counts[bin_idx] += 1
    
    # Expected count per bin for uniform distribution
    expected_per_bin = n_points / (n_bins ** dimensions)
    
    # Compute variance of bin counts (simplified discrepancy measure)
    variance = np.var(bin_counts)
    
    # Normalize by expected count
    normalized_discrepancy = np.sqrt(variance) / expected_per_bin if expected_per_bin > 0 else 0
    
    return normalized_discrepancy


def compute_minimum_distance(points: np.ndarray) -> float:
    """Compute minimum distance between any two points."""
    n_points = len(points)
    min_dist = float('inf')
    
    # Sample-based approximation for efficiency
    sample_size = min(100, n_points)
    sample_indices = np.random.choice(n_points, sample_size, replace=False)
    
    for i in sample_indices:
        for j in range(n_points):
            if i != j:
                dist = np.linalg.norm(points[i] - points[j])
                if dist < min_dist:
                    min_dist = dist
    
    return min_dist


def compute_coverage_metrics(points: np.ndarray) -> dict:
    """Compute various metrics of parameter space coverage."""
    n_bins_list = [5, 10, 20]
    discrepancies = [compute_discrepancy(points, n_bins) for n_bins in n_bins_list]
    
    # Compute empty bins
    n_bins = 10
    dimensions = points.shape[1]
    bin_counts = np.zeros([n_bins] * dimensions)
    
    for point in points:
        bin_idx = tuple(min(int(p * n_bins), n_bins - 1) for p in point)
        bin_counts[bin_idx] += 1
    
    empty_bins = np.sum(bin_counts == 0)
    total_bins = n_bins ** dimensions
    
    return {
        'discrepancy_5': discrepancies[0],
        'discrepancy_10': discrepancies[1],
        'discrepancy_20': discrepancies[2],
        'empty_bins': int(empty_bins),
        'total_bins': int(total_bins),
        'coverage_percent': 100 * (1 - empty_bins / total_bins),
        'min_distance': compute_minimum_distance(points)
    }


def visualize_ascii_2d(points: np.ndarray, width: int = 40, height: int = 20, title: str = ""):
    """Create ASCII visualization of 2D point distribution."""
    if points.shape[1] != 2:
        print("Error: Can only visualize 2D points")
        return
    
    # Create grid
    grid = [[' ' for _ in range(width)] for _ in range(height)]
    
    # Map points to grid
    for point in points:
        x = int(point[0] * (width - 1))
        y = int(point[1] * (height - 1))
        x = max(0, min(x, width - 1))
        y = max(0, min(y, height - 1))
        grid[height - 1 - y][x] = '●'
    
    # Print visualization
    if title:
        print(f"\n{title}")
        print("─" * width)
    
    for row in grid:
        print(''.join(row))
    
    print("─" * width)


def run_comparison(n_points: int = 100):
    """Run comparison between Monte Carlo and QMC sampling."""
    print("\n" + "="*70)
    print(" QMC vs Monte Carlo: Parameter Space Coverage Demonstration")
    print("="*70)
    
    print(f"\nGenerating {n_points} points in 2D parameter space...")
    
    # Generate both types of points
    mc_points = generate_monte_carlo_points(n_points)
    qmc_points = generate_halton_points(n_points)
    
    # Compute metrics
    print("\nComputing coverage metrics...")
    mc_metrics = compute_coverage_metrics(mc_points)
    qmc_metrics = compute_coverage_metrics(qmc_points)
    
    # Display visualizations
    visualize_ascii_2d(mc_points, width=50, height=25, 
                       title="Monte Carlo (Pseudo-Random) Sampling")
    
    visualize_ascii_2d(qmc_points, width=50, height=25, 
                       title="QMC (Halton Sequence) Sampling")
    
    # Display metrics comparison
    print("\n" + "="*70)
    print(" Coverage Metrics Comparison")
    print("="*70)
    print(f"\n{'Metric':<30} {'Monte Carlo':>18} {'QMC (Halton)':>18}")
    print("─"*70)
    print(f"{'Discrepancy (5 bins):':<30} {mc_metrics['discrepancy_5']:>18.4f} {qmc_metrics['discrepancy_5']:>18.4f}")
    print(f"{'Discrepancy (10 bins):':<30} {mc_metrics['discrepancy_10']:>18.4f} {qmc_metrics['discrepancy_10']:>18.4f}")
    print(f"{'Discrepancy (20 bins):':<30} {mc_metrics['discrepancy_20']:>18.4f} {qmc_metrics['discrepancy_20']:>18.4f}")
    print(f"{'Empty bins (out of 100):':<30} {mc_metrics['empty_bins']:>18} {qmc_metrics['empty_bins']:>18}")
    print(f"{'Coverage percentage:':<30} {mc_metrics['coverage_percent']:>17.1f}% {qmc_metrics['coverage_percent']:>17.1f}%")
    print(f"{'Min distance:':<30} {mc_metrics['min_distance']:>18.4f} {qmc_metrics['min_distance']:>18.4f}")
    print("─"*70)
    
    # Compute improvements
    disc_improvement = (mc_metrics['discrepancy_10'] - qmc_metrics['discrepancy_10']) / mc_metrics['discrepancy_10'] * 100
    coverage_improvement = qmc_metrics['coverage_percent'] - mc_metrics['coverage_percent']
    empty_reduction = (mc_metrics['empty_bins'] - qmc_metrics['empty_bins']) / mc_metrics['empty_bins'] * 100 if mc_metrics['empty_bins'] > 0 else 0
    
    print(f"\n{'QMC Improvements:'}")
    print("─"*70)
    print(f"  • Discrepancy reduction: {disc_improvement:.1f}%")
    print(f"  • Coverage improvement: +{coverage_improvement:.1f}%")
    print(f"  • Empty bins reduction: {empty_reduction:.1f}%")
    print("─"*70)
    
    # Interpretation
    print(f"\n{'Interpretation:'}")
    print("─"*70)
    print("Lower discrepancy = More uniform coverage")
    print("Higher coverage = Fewer gaps in parameter space")
    print("Smaller min distance = Risk of redundant sampling")
    print()
    print("QMC (Halton) sequences provide systematic, uniform coverage while")
    print("Monte Carlo sampling shows clustering and gaps due to randomness.")
    print()
    print("For Pollard's Rho, this means:")
    print("  ✓ QMC explores parameter space more efficiently")
    print("  ✓ Fewer redundant (c, x0) pairs tested")
    print("  ✓ More consistent performance across runs")
    print("  ✓ Reduced variance in iteration counts")
    print("="*70 + "\n")


def demonstrate_convergence():
    """Demonstrate how QMC converges faster than Monte Carlo."""
    print("\n" + "="*70)
    print(" Convergence Rate Demonstration")
    print("="*70)
    
    print("\nAs we add more points, how does coverage improve?\n")
    
    point_counts = [10, 25, 50, 100, 200]
    
    print(f"{'N Points':<12} {'MC Discrepancy':>18} {'QMC Discrepancy':>18} {'Improvement':>16}")
    print("─"*70)
    
    for n in point_counts:
        mc_points = generate_monte_carlo_points(n, seed=42)
        qmc_points = generate_halton_points(n)
        
        mc_disc = compute_discrepancy(mc_points, n_bins=10)
        qmc_disc = compute_discrepancy(qmc_points, n_bins=10)
        
        improvement = (mc_disc - qmc_disc) / mc_disc * 100
        
        print(f"{n:<12} {mc_disc:>18.4f} {qmc_disc:>18.4f} {improvement:>15.1f}%")
    
    print("─"*70)
    print("\nNote: QMC maintains better uniformity even with fewer points,")
    print("while Monte Carlo requires many more points to achieve similar coverage.")
    print("="*70 + "\n")


def main():
    """Run all demonstrations."""
    print("\n" + "="*70)
    print(" Quasi-Monte Carlo (QMC) Conceptual Demonstration")
    print(" Showing Why QMC Reduces Variance in Pollard's Rho")
    print("="*70)
    
    print("\nThis demonstration shows how QMC low-discrepancy sequences")
    print("provide better parameter space coverage than random sampling.")
    print("\nIn Pollard's Rho factorization, parameters (c, x0) determine")
    print("the pseudo-random sequence. Better coverage means:")
    print("  • More systematic exploration of parameter space")
    print("  • Reduced likelihood of missing good parameter combinations")
    print("  • More consistent performance (lower variance)")
    
    # Run main comparison
    run_comparison(n_points=100)
    
    # Show convergence properties
    demonstrate_convergence()
    
    print("Conclusion:")
    print("─"*70)
    print("QMC sequences (like Halton) fill parameter space more uniformly")
    print("than pseudo-random sampling, leading to:")
    print("  1. Better exploration efficiency")
    print("  2. Reduced redundancy in parameter selection")
    print("  3. Lower variance in algorithm performance")
    print()
    print("This is why applying QMC to Pollard's Rho parameter selection")
    print("reduces variance while maintaining competitive mean performance.")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
