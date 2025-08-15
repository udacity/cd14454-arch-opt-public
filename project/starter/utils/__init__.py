"""
PneumoniaMNIST Utilities Package

A comprehensive toolkit for medical imaging binary classification with PyTorch.
Designed specifically for the PneumoniaMNIST dataset and clinical deployment scenarios.

This package provides end-to-end utilities for:
    - Medical imaging data loading and preprocessing
    - Model creation and training with clinical focus
    - Performance profiling and optimization for deployment
    - Comprehensive evaluation with clinical metrics
    - Advanced model optimization techniques
    - Clinical visualization and analysis tools

Project: Efficient Medical Diagnostics with Hardware-Aware AI
"""

# Package metadata
__version__ = "1.0.0"
__author__ = "Udacity Instructor"
__project__ = "Efficient Medical Diagnostics with Hardware-Aware AI"

# =============================================================================
# DATA LOADING AND PREPROCESSING
# =============================================================================

from .data_loader import (
    # Core data loading functions
    load_pneumoniamnist,
    get_sample_batch,
    get_dataset_info,
    
    # Dataset analysis utilities
    explore_dataset_splits,
    
    # Visualization functions
    visualize_sample_images,
    
    # Clinical constants
    PNEUMONIA_CLASSES
)

# =============================================================================
# MODEL CREATION AND TRAINING
# =============================================================================

from .model import (
    # Model architecture classes
    ResNetBaseline,
    
    # Model creation utilities
    create_baseline_model,
    get_model_info,
    count_parameters_by_type,
    
    # Training functions
    train_baseline_model,
    plot_training_history
)

# =============================================================================
# PERFORMANCE PROFILING AND ANALYSIS
# =============================================================================

from .profiling import (
    # Core profiling classes
    PerformanceProfiler,
    
    # Environment and hardware utilities
    get_gpu_info,
    check_environment,
    
    # Timing utilities
    measure_time
)

# =============================================================================
# MODEL EVALUATION AND CLINICAL METRICS
# =============================================================================

from .evaluation import (
    # Core evaluation classes
    ClassificationEvaluator,
    
    # Evaluation functions
    evaluate_model,
    evaluate_with_multiple_thresholds,
    
    # Threshold optimization
    find_optimal_threshold,
    
    # Result formatting
    format_evaluation_results
)

# =============================================================================
# VISUALIZATION AND ANALYSIS
# =============================================================================

from .visualization import (
    # Dataset visualization
    plot_dataset_distribution,
    
    # Performance visualization
    plot_performance_profile,
    plot_operation_breakdown,
    plot_batch_size_comparison
)

# =============================================================================
# MODEL OPTIMIZATION TECHNIQUES
# =============================================================================

from .architecture_optimization import (
    # Individual optimization techniques
    apply_interpolation_removal_optimization,
    apply_depthwise_separable_optimization,
    apply_grouped_convolution_optimization,
    apply_inverted_residual_optimization,
    apply_lowrank_factorization,
    apply_channel_optimization,
    apply_parameter_sharing,
    create_optimized_model
)

# =============================================================================
# PACKAGE UTILITIES AND HELPERS
# =============================================================================

def get_package_info():
    """
    Get comprehensive package information for documentation and debugging.
    
    Returns:
        Dictionary containing package metadata, available utilities, and usage tips
    """
    return {
        'package': 'pneumoniamnist_utils',
        'version': __version__,
        'author': __author__,
        'project': __project__,
        'description': 'Clinical AI utilities for PneumoniaMNIST binary classification',
        
        'modules': {
            'data_loader': 'Medical imaging data loading and preprocessing',
            'model': 'ResNet baseline model creation and training',
            'profiling': 'Performance profiling for clinical deployment',
            'evaluation': 'Clinical evaluation metrics and analysis',
            'visualization': 'Medical imaging visualization utilities',
            'architecture_optimization': 'Hardware-aware model optimization techniques'
        },
        
        'key_features': [
            'Clinical-focused medical imaging pipeline',
            'Hardware-aware optimization for edge deployment',
            'Comprehensive performance profiling',
            'Clinical evaluation metrics (sensitivity, specificity)',
            'Real-time inference optimization',
            'Educational framework for computer vision learning'
        ]
    }