
"""
Visualization utilities for displaying performance metrics, model analysis results, and comparative charts.

This module provides comprehensive visualization functions for analyzing model performance,
dataset distributions, timing measurements, and operation breakdowns. Designed for the
PneumoniaMNIST binary classification project as part of Udacity coursework.
"""


import torch
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Any, Dict, List, Optional, Tuple, Union

# Set consistent styling for all plots
plt.style.use('default')
sns.set_palette("husl")


def plot_dataset_distribution(dataset_stats: Dict[str, Dict[str, Union[int, Dict[str, int]]]]) -> None:
    """
    Visualize dataset split distribution and class balance for binary classification.
    
    Creates a two-panel visualization showing:
    1. Total samples per dataset split (train/val/test)
    2. Class distribution breakdown with percentages
    
    Args:
        dataset_stats: Dictionary from data_utils.explore_dataset_splits() containing:
            - Keys: 'train', 'val', 'test', optionally 'summary'
            - Values: Dict with 'total', 'class_stats' containing normal/pneumonia counts
    
    Returns:
        None: Displays the plots directly
        
    Example:
        >>> stats = explore_dataset_splits(train_loader, val_loader, test_loader)
        >>> plot_dataset_distribution(stats)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Extract dataset splits (exclude summary if present)
    splits = [k for k in dataset_stats.keys() if k != 'summary']
    colors = ['#3498db', '#e74c3c', '#2ecc71']  # Blue, red, green

    # Plot 1: Dataset split sizes
    totals = [dataset_stats[split]["total"] for split in splits]
    
    bars1 = ax1.bar(splits, totals, color=colors, alpha=0.7, edgecolor='black')
    ax1.set_title('Dataset Split Sizes', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Number of Samples')
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels on top of bars
    for bar, total in zip(bars1, totals):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
                f'{total:,}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Class distribution with grouped bars
    width = 0.35
    x = np.arange(len(splits))
    
    normal_counts = [dataset_stats[split]["class_stats"]["normal"] for split in splits]
    pneumonia_counts = [dataset_stats[split]["class_stats"]["pneumonia"] for split in splits]
    
    bars2 = ax2.bar(x - width/2, normal_counts, width, label='Normal', 
                   color='#3498db', alpha=0.7, edgecolor='black')
    bars3 = ax2.bar(x + width/2, pneumonia_counts, width, label='Pneumonia', 
                   color='#e74c3c', alpha=0.7, edgecolor='black')
    
    ax2.set_title('Class Distribution by Split', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Number of Samples')
    ax2.set_xlabel('Dataset Split')
    ax2.set_xticks(x)
    ax2.set_xticklabels(splits)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # Add percentage labels for clinical context
    for i, split in enumerate(splits):
        total = dataset_stats[split]["total"]
        normal_pct = normal_counts[i] / total * 100
        pneumonia_pct = pneumonia_counts[i] / total * 100
        
        ax2.text(i - width/2, normal_counts[i] + 50, f'{normal_pct:.1f}%', 
                ha='center', va='bottom', fontsize=10)
        ax2.text(i + width/2, pneumonia_counts[i] + 50, f'{pneumonia_pct:.1f}%', 
                ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.show()


def plot_performance_profile(timing_results: Dict[str, float]) -> None:
    """
    Visualize comprehensive performance profiling measurements for model inference timing.
    
    Creates a four-panel dashboard showing:
    1. Core timing statistics (mean, min, max, etc.)
    2. Single vs batch performance comparison with efficiency metrics
    3. Throughput analysis for different processing modes
    4. Timing distribution visualization with target benchmarks
    
    Args:
        timing_results: Dictionary containing timing measurements with keys:
            - 'mean_ms', 'min_ms', 'max_ms', 'median_ms', 'std_ms', 'p95_ms'
            - 'single_sample_ms', 'batch_total_ms', 'batch_size'
            - 'throughput_samples_per_sec', 'batch_throughput_samples_per_sec'
            - 'batch_efficiency'
    
    Returns:
        None: Displays the comprehensive performance dashboard
        
    Example:
        >>> profiler = PerformanceProfiler()
        >>> results = profiler.profile_inference_time(model, input_tensor)
        >>> plot_performance_profile(results)
    """
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Core Timing Statistics
    timing_metrics = ['mean_ms', 'min_ms', 'max_ms', 'median_ms', 'std_ms', 'p95_ms']
    timing_values = [timing_results[metric] for metric in timing_metrics]
    timing_labels = ['Mean', 'Min', 'Max', 'Median', 'Std Dev', '95th %ile']
    
    bars1 = ax1.bar(timing_labels, timing_values, color='#3498db', alpha=0.7, edgecolor='black')
    ax1.set_title('Single Sample Inference Time Statistics', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Time (ms)')
    ax1.set_ylim(0, max(timing_values) * 1.15)
    ax1.grid(axis='y', alpha=0.3)
    
    # Add value labels for precise readings
    for bar, value in zip(bars1, timing_values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(timing_values)*0.01,
                f'{value:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Single vs Batch Performance Comparison
    batch_size = timing_results.get('batch_size', 1)
    single_latency = timing_results['single_sample_ms']
    batch_total = timing_results.get('batch_total_ms', single_latency * batch_size)
    batch_efficiency = timing_results.get('batch_efficiency', 1.0)
    
    latency_labels = ['Single Sample\nLatency', 'Batch Total\nLatency']
    latency_values = [single_latency, batch_total]
    latency_colors = ['#3498db', '#e74c3c']
    
    bars2 = ax2.bar(latency_labels, latency_values, color=latency_colors, alpha=0.7, edgecolor='black')
    ax2.set_title(f'Single vs Batch Performance (Batch Size: {batch_size})', fontweight='bold')
    ax2.set_ylabel('Time (ms)', color='black')
    ax2.set_ylim(0, max(latency_values) * 1.15)
    ax2.grid(axis='y', alpha=0.3)
    
    # Add latency labels
    for bar, value in zip(bars2, latency_values):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(latency_values)*0.02,
                f'{value:.2f}ms', ha='center', va='center', fontweight='bold')
    
    # Add batch efficiency indicator
    efficiency_text = f"Batch Efficiency: {batch_efficiency:.2f}x speedup"
    efficiency_color = '#2ecc71' if batch_efficiency > 1.0 else '#e74c3c'
    ax2.text(0.5, 0.55, efficiency_text, transform=ax2.transAxes, 
             ha='center', va='top', fontsize=12, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.3', facecolor=efficiency_color, alpha=0.8))
    
    # Plot 3: Throughput Analysis
    single_throughput = timing_results.get('throughput_samples_per_sec', 0)
    batch_throughput = timing_results.get('batch_throughput_samples_per_sec', single_throughput)
    
    throughput_labels = ['Single Sample\nThroughput', 'Batch\nThroughput']
    throughput_values = [single_throughput, batch_throughput]
    throughput_colors = ['#9b59b6', '#f39c12']
    
    bars3 = ax3.bar(throughput_labels, throughput_values, color=throughput_colors, alpha=0.7, edgecolor='black')
    ax3.set_title('Throughput Comparison', fontweight='bold')
    ax3.set_ylabel('Samples per Second')
    ax3.set_ylim(0, max(throughput_values) * 1.15)
    ax3.grid(axis='y', alpha=0.3)
    
    # Add throughput value labels
    for bar, value in zip(bars3, throughput_values):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(throughput_values)*0.02,
                f'{value:.0f}', ha='center', va='bottom', fontweight='bold')
    
    # Plot 4: Timing Distribution Visualization with Benchmarks
    mean_ms = timing_results['mean_ms']
    std_ms = timing_results['std_ms']
    p95_ms = timing_results['p95_ms']
    p99_ms = timing_results.get('p99_ms', p95_ms)
    
    # Generate synthetic distribution for visualization
    np.random.seed(42)
    timing_samples = np.random.normal(mean_ms, std_ms, 1000)
    timing_samples = np.clip(timing_samples, timing_results['min_ms'], timing_results['max_ms'])
    
    ax4.hist(timing_samples, bins=30, alpha=0.7, color='lightblue', edgecolor='black', density=True)
    ax4.axvline(mean_ms, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_ms:.2f}ms')
    ax4.axvline(p95_ms, color='orange', linestyle='--', linewidth=2, label=f'95th %ile: {p95_ms:.2f}ms')
    ax4.axvline(p99_ms, color='purple', linestyle='--', linewidth=2, label=f'99th %ile: {p99_ms:.2f}ms')
    
    # Add clinical target benchmark (10ms for real-time inference)
    target_latency = 10.0
    if mean_ms > target_latency:
        ax4.axvline(target_latency, color='green', linestyle=':', linewidth=3, 
                   label=f'Target: {target_latency}ms', alpha=0.8)
    
    ax4.set_title('Inference Time Distribution', fontweight='bold')
    ax4.set_xlabel('Time (ms)')
    ax4.set_ylabel('Density')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    plt.show()


def plot_operation_breakdown(op_breakdown: Dict[str, Any]) -> None:
    """
    Visualize PyTorch profiler operation breakdown with enhanced readability.
    
    Creates a dual visualization showing:
    1. Horizontal bar chart of operation percentages
    2. Pie chart with external labels for operation distribution
    
    Args:
        op_breakdown: Dictionary with operation names as keys and percentage 
                     values as values from PyTorch profiler analysis
    
    Returns:
        None: Displays the operation breakdown visualization
        
    Note:
        Filters operations to show only those consuming >2% of total time
        for cleaner visualization and better insights.
        
    Example:
        >>> profiler = PerformanceProfiler()
        >>> results = profiler.profile_with_pytorch_profiler(model, input_tensor)
        >>> plot_operation_breakdown(results['operation_breakdown'])
    """
    # Filter operations > 2% for cleaner visualization
    significant_ops = {op: pct for op, pct in op_breakdown.items() if pct > 2.0}
    
    if not significant_ops:
        print("⚠️ No significant operations found (all < 2%)")
        return
    
    # Sort by percentage for better readability
    sorted_ops = sorted(significant_ops.items(), key=lambda x: x[1], reverse=True)
    operations = [op.replace('_', ' ').title() for op, _ in sorted_ops]
    percentages = [pct for _, pct in sorted_ops]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Plot 1: Horizontal bar chart for precise comparison
    colors = plt.cm.Set3(np.linspace(0, 1, len(operations)))
    bars = ax1.barh(operations, percentages, color=colors, alpha=0.8, edgecolor='black')
    ax1.set_title('Operation Time Breakdown', fontweight='bold', fontsize=14)
    ax1.set_xlabel('Percentage of Total Time (%)')
    ax1.grid(axis='x', alpha=0.3)
    ax1.set_xlim(0, max(percentages) * 1.1)  # Add space for labels
    
    # Add percentage labels for precise readings
    for bar, pct in zip(bars, percentages):
        ax1.text(bar.get_width() + max(percentages)*0.01, bar.get_y() + bar.get_height()/2,
                f'{pct:.1f}%', ha='left', va='center', fontweight='bold')
    
    # Plot 2: Pie chart with enhanced labeling
    pie_colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']
    
    wedges, texts = ax2.pie(percentages, labels=None, colors=pie_colors[:len(operations)], 
                           startangle=90, wedgeprops=dict(edgecolor='white', linewidth=2))
    
    ax2.set_title('Operation Distribution', fontweight='bold', fontsize=14)
    
    # Add external labels with better positioning to avoid overlap
    bbox_props = dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8, edgecolor="black")
    
    for i, (wedge, op, pct) in enumerate(zip(wedges, operations, percentages)):
        # Calculate label position at middle of wedge
        angle = (wedge.theta2 + wedge.theta1) / 2
        
        # Position labels outside the pie with smart positioning
        x = 1.3 * np.cos(np.radians(angle))
        y = 1.3 * np.sin(np.radians(angle))
        
        # Force pooling operations to left side to prevent label overlap
        if 'Pooling' in op:
            x = -abs(x) 
            ha = 'right'
        else:
            ha = 'left' if x > 0 else 'right'
        
        # Add label with background for better readability
        ax2.annotate(f'{op}\n{pct:.1f}%', 
                    xy=(np.cos(np.radians(angle)), np.sin(np.radians(angle))), 
                    xytext=(x, y),
                    ha=ha, va='center',
                    fontsize=10, fontweight='bold',
                    bbox=bbox_props,
                    arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # Expand plot area to accommodate external labels
    ax2.set_xlim(-2, 2)
    ax2.set_ylim(-2, 2)
    
    plt.tight_layout()
    plt.show()


def plot_batch_size_comparison(batch_results: Dict[str, Dict[str, Any]]) -> None:
    """
    Visualize comprehensive performance analysis across different batch sizes.
    
    Creates a four-panel dashboard showing:
    1. Latency scaling with batch size
    2. Throughput optimization curves
    3. Batch processing efficiency trends
    4. Summary table with key metrics
    
    Args:
        batch_results: Dictionary with batch size results from profile_multiple_batch_sizes():
            - Keys: 'batch_1', 'batch_8', 'batch_16', etc.
            - Values: Dict containing 'timing' and 'memory' results
    
    Returns:
        None: Displays the comprehensive batch size analysis
        
    Example:
        >>> profiler = PerformanceProfiler()
        >>> results = profiler.profile_multiple_batch_sizes(model, input_shape)
        >>> plot_batch_size_comparison(results)
    """
    if not batch_results:
        print("No batch results available for visualization")
        return
    
    # Extract valid results (filter out errors)
    valid_results = {k: v for k, v in batch_results.items() if 'error' not in v}
    
    if not valid_results:
        print("No valid batch results for visualization")
        return
    
    # Extract performance data for plotting
    batch_sizes = [int(k.split('_')[1]) for k in valid_results.keys()]
    single_latencies = [v['timing']['single_sample_ms'] for v in valid_results.values()]
    batch_latencies = [v['timing']['batch_total_ms'] for v in valid_results.values()]
    throughputs = [v['timing']['batch_throughput_samples_per_sec'] for v in valid_results.values()]
    efficiencies = [v['timing']['batch_efficiency'] for v in valid_results.values()]
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Latency vs Batch Size (log scale for better visualization)
    ax1.plot(batch_sizes, single_latencies, 'o-', color='#3498db', linewidth=2, 
             markersize=8, label='Per Sample Latency')
    ax1.plot(batch_sizes, batch_latencies, 's-', color='#e74c3c', linewidth=2, 
             markersize=8, label='Total Batch Latency')
    ax1.set_xlabel('Batch Size')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('Latency vs Batch Size', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_xscale('log', base=2)  # Log scale for better trend visibility
    
    # Plot 2: Throughput Optimization Curve
    ax2.plot(batch_sizes, throughputs, 'o-', color='#2ecc71', linewidth=2, markersize=8)
    ax2.set_xlabel('Batch Size')
    ax2.set_ylabel('Throughput (samples/sec)')
    ax2.set_title('Throughput vs Batch Size', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.set_xscale('log', base=2)
    
    # Plot 3: Batch Processing Efficiency
    ax3.plot(batch_sizes, efficiencies, 'o-', color='#9b59b6', linewidth=2, markersize=8)
    ax3.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='No Efficiency Gain')
    ax3.set_xlabel('Batch Size')
    ax3.set_ylabel('Efficiency Factor (speedup)')
    ax3.set_title('Batch Processing Efficiency', fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_xscale('log', base=2)
    
    # Plot 4: Performance Summary Table
    ax4.axis('tight')
    ax4.axis('off')
    
    # Create comprehensive summary table
    table_data = []
    for i, batch_size in enumerate(batch_sizes):
        table_data.append([
            f'{batch_size}',
            f'{single_latencies[i]:.2f}ms',
            f'{throughputs[i]:.0f} sps',
            f'{efficiencies[i]:.2f}x'
        ])
    
    table = ax4.table(cellText=table_data,
                     colLabels=['Batch Size', 'Per Sample\nLatency', 'Throughput', 'Efficiency'],
                     cellLoc='center',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Style the table for better readability
    for i in range(len(table_data) + 1):
        for j in range(4):
            cell = table[(i, j)]
            if i == 0:  # Header row
                cell.set_facecolor('#3498db')
                cell.set_text_props(weight='bold', color='white')
            else:  # Data rows with alternating colors
                cell.set_facecolor('#f8f9fa' if i % 2 == 0 else 'white')
    
    ax4.set_title('Batch Size Performance Summary', fontweight='bold', pad=20)
    
    plt.suptitle('Batch Size Analysis', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()