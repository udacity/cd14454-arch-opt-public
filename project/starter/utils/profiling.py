
"""
Performance profiling utilities for comprehensive model optimization analysis.

This module provides core measurement functionality for PyTorch model performance
analysis without conducting the analysis itself. Includes timing measurements,
FLOPs calculation, memory profiling, and PyTorch profiler integration.

Designed for the PneumoniaMNIST binary classification project as part of Udacity coursework.
Focuses on clinical deployment requirements where inference efficiency is critical.
"""

import platform
import sys
import time
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union
import psutil

import numpy as np
from fvcore.nn import flop_count
from fvcore.nn.flop_count import _DEFAULT_SUPPORTED_OPS

import torch
import torch.nn as nn
import torch.profiler
from torch.cuda.amp import autocast


class PerformanceProfiler:
    """
    Comprehensive profiler for measuring and analyzing model performance metrics.
    
    Provides standardized methods for timing analysis, FLOPs calculation, memory profiling,
    and detailed operation breakdown. Designed for clinical AI applications where
    performance optimization is critical for deployment feasibility.
    
    Attributes:
        device: Target device for profiling ('cuda' or 'cpu')
        is_cuda: Boolean indicating if CUDA is being used
        use_amp: Whether to enable automatic mixed precision
    """
    
    def __init__(self, device: str = "auto", use_amp: bool = False) -> None:
        """
        Initialize the performance profiler with device and optimization settings.
        
        Args:
            device: Target device ('cuda', 'cpu', or 'auto' for automatic selection)
            use_amp: Enable automatic mixed precision for supported CUDA devices
        """
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        
        self.device = device
        self.is_cuda = "cuda" in device.lower()
        self.use_amp = use_amp
        
    def profile_inference_time(self, model: nn.Module, input_tensor: torch.Tensor,
                            num_runs: int = 100, warmup_runs: int = 10) -> Dict[str, Union[float, int, Dict, None]]:
        """
        Profile comprehensive inference timing with single-sample and batch analysis.
        
        Measures both single-sample latency (critical for real-time clinical applications)
        and batch processing efficiency. Includes statistical analysis with percentiles
        and efficiency metrics for deployment planning.
        
        Args:
            model: PyTorch model to profile
            input_tensor: Input tensor with any batch size for profiling
            num_runs: Number of timing measurements for statistical accuracy
            warmup_runs: Number of warmup iterations to stabilize GPU performance
            
        Returns:
            Comprehensive timing dictionary containing:
                - Single-sample metrics: mean_ms, std_ms, min_ms, max_ms, etc.
                - Batch metrics: batch_total_ms, batch_efficiency, throughput
                - Statistical analysis: p95_ms, p99_ms for SLA planning
                - Detailed breakdowns for both single and batch processing
        
        Example:
            >>> profiler = PerformanceProfiler(device='cuda', use_amp=True)
            >>> model = create_baseline_model()
            >>> input_tensor = torch.randn(8, 3, 224, 224)
            >>> results = profiler.profile_inference_time(model, input_tensor)
            >>> print(f"Single sample latency: {results['single_sample_ms']:.2f}ms")
        """
        model.eval()
        input_tensor = input_tensor.to(self.device)
        original_batch_size = input_tensor.size(0)
        
        def _time_inference(input_data: torch.Tensor, runs: int = num_runs) -> np.ndarray:
            """Internal helper function for precise timing measurements."""
            times = []
            
            # Warmup phase - essential for GPU performance stabilization
            with torch.no_grad():
                for _ in range(warmup_runs):
                    if self.is_cuda:
                        torch.cuda.synchronize()  # Ensure GPU operations complete
                    _ = call_model(model, input_data, self.use_amp)
                    if self.is_cuda:
                        torch.cuda.synchronize()
            
            # Actual timing measurements
            with torch.no_grad():
                for _ in range(runs):
                    if self.is_cuda:
                        torch.cuda.synchronize()
                    
                    start_time = time.perf_counter()
                    output = call_model(model, input_data, self.use_amp)
                    
                    if self.is_cuda:
                        torch.cuda.synchronize()  # Critical for accurate GPU timing
                    
                    end_time = time.perf_counter()
                    times.append((end_time - start_time) * 1000)  # Convert to milliseconds
            
            return np.array(times)
        
        # Measure true single-sample latency (batch_size=1)
        single_input = input_tensor[:1]
        single_times = _time_inference(single_input)
        
        # Measure batch latency if original batch_size > 1
        if original_batch_size > 1:
            batch_times = _time_inference(input_tensor)
        else:
            batch_times = single_times  # Same measurement for batch_size=1
        
        # Calculate comprehensive statistical metrics
        single_stats = {
            'mean': float(np.mean(single_times)),
            'std': float(np.std(single_times)),
            'min': float(np.min(single_times)),
            'max': float(np.max(single_times)),
            'median': float(np.median(single_times)),
            'p95': float(np.percentile(single_times, 95)),  # Critical for SLA planning
            'p99': float(np.percentile(single_times, 99))   # Worst-case performance
        }
        
        batch_stats = {
            'mean': float(np.mean(batch_times)),
            'std': float(np.std(batch_times)),
            'min': float(np.min(batch_times)),
            'max': float(np.max(batch_times)),
            'median': float(np.median(batch_times)),
            'p95': float(np.percentile(batch_times, 95)),
            'p99': float(np.percentile(batch_times, 99))
        }
        
        # Calculate batch processing efficiency (how much batching helps)
        batch_efficiency = single_stats['mean'] / (batch_stats['mean'] / original_batch_size) if original_batch_size > 1 else 1.0
        
        return {
            # Primary metrics (backward compatibility)
            'mean_ms': single_stats['mean'],
            'std_ms': single_stats['std'],
            'min_ms': single_stats['min'],
            'max_ms': single_stats['max'],
            'median_ms': single_stats['median'],
            'p95_ms': single_stats['p95'],
            'p99_ms': single_stats['p99'],
            'single_sample_ms': single_stats['mean'],
            'throughput_samples_per_sec': 1000 / single_stats['mean'],
            
            # Batch processing analysis
            'batch_size': original_batch_size,
            'batch_total_ms': batch_stats['mean'],
            'batch_per_sample_ms': batch_stats['mean'] / original_batch_size,
            'batch_throughput_samples_per_sec': original_batch_size * 1000 / batch_stats['mean'],
            'batch_efficiency': batch_efficiency,
            
            # Detailed statistical breakdowns
            'single_sample_stats': single_stats,
            'batch_stats': batch_stats if original_batch_size > 1 else None,
        }

    def profile_flops(self, model: nn.Module, input_tensor: torch.Tensor, 
                     memory_format: torch.memory_format = torch.preserve_format) -> Dict[str, Any]:
        """
        Profile Floating Point Operations (FLOPs) with detailed module breakdown.
        
        Calculates computational complexity in GFLOPS (billion floating point operations)
        and provides per-module analysis for optimization targeting. Essential for
        understanding computational requirements and deployment feasibility.
        
        Args:
            model: PyTorch model to analyze
            input_tensor: Representative input tensor for FLOP calculation
            memory_format: Memory layout format for tensor operations
            
        Returns:
            Dictionary containing:
                - total_gflops: Total computational load in GFLOPS
                - gflops_per_sample: GFLOPS per individual sample
                - module_breakdown_gflops: Per-module FLOP analysis
                - module_percentage: Relative computational load per module
                - supported_ops_count: Number of successfully profiled operations
        
        Note:
            Some custom operations may not be supported by fvcore and will not
            contribute to the FLOP count. Check supported_ops_count for coverage.
            
        Example:
            >>> profiler = PerformanceProfiler()
            >>> model = create_baseline_model()
            >>> input_tensor = torch.randn(1, 3, 224, 224)
            >>> flop_results = profiler.profile_flops(model, input_tensor)
            >>> print(f"Model complexity: {flop_results['gflops_per_sample']:.2f} GFLOPS/sample")
        """
        try:
            model.eval()
            input_tensor = input_tensor.to(self.device, memory_format=memory_format)
            batch_size = input_tensor.size(0)
            
            # Calculate FLOPs with comprehensive module breakdown
            with torch.no_grad():
                flops_dict, _ = flop_count(
                    model, 
                    (input_tensor,),
                    supported_ops=_DEFAULT_SUPPORTED_OPS
                )
                
                # Process and organize module breakdown
                module_flops = {}
                total_flops = 0
                
                for module_name, flops in flops_dict.items():
                    if flops > 0:  # Only include modules with computational load
                        module_flops[module_name] = flops
                        total_flops += flops
                
                # Sort modules by computational load for optimization prioritization
                sorted_modules = dict(sorted(module_flops.items(), 
                                           key=lambda x: x[1], reverse=True))
            
            # Calculate per-sample metrics (critical for deployment planning)
            flops_per_sample = total_flops / batch_size if batch_size > 0 else total_flops
            
            # Calculate percentage breakdown for top computational modules
            percentage_breakdown = {}
            if total_flops > 0:
                percentage_breakdown = {
                    name: (flops / total_flops) * 100 
                    for name, flops in list(sorted_modules.items())[:10]  # Top 10 modules
                }
            
            result = {
                'total_gflops': total_flops,
                'gflops_per_sample': flops_per_sample,
                'batch_size': batch_size,
                'module_breakdown_gflops': sorted_modules,
                'module_percentage': percentage_breakdown,
                'supported_ops_count': len([f for f in flops_dict.values() if f > 0])
            }
            
            return result
            
        except Exception as e:
            return {
                'error': f'FLOPs calculation failed: {str(e)}',
                'suggestion': 'Try with a simpler model or check if all operations are supported'
            }
    
    def profile_memory_usage(self, model: nn.Module, input_tensor: torch.Tensor) -> Dict[str, Any]:
        """
        Profile GPU memory usage during model inference.
        
        Provides detailed memory analysis including model parameters, activations,
        and peak memory consumption. Critical for deployment planning and batch
        size optimization in resource-constrained environments.
        
        Args:
            model: PyTorch model to profile
            input_tensor: Representative input for memory profiling
            
        Returns:
            Dictionary containing:
                - baseline_memory_mb: Initial GPU memory usage
                - peak_memory_mb: Maximum memory during inference
                - memory_increase_mb: Additional memory required for inference
                - component_breakdown: Detailed breakdown of memory consumers
        
        Note:
            Only available for CUDA devices. Memory profiling helps optimize
            batch sizes and identify memory bottlenecks for deployment.
            
        Example:
            >>> profiler = PerformanceProfiler(device='cuda')
            >>> memory_results = profiler.profile_memory_usage(model, input_tensor)
            >>> print(f"Peak memory: {memory_results['peak_memory_mb']:.1f} MB")
        """
        if not self.is_cuda:
            return {'error': 'GPU memory profiling only available with CUDA'}
        
        try:
            # Reset memory statistics for clean measurement
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            
            # Measure baseline memory usage
            baseline_memory = torch.cuda.memory_allocated() / (1024 * 1024)  # Convert to MB
            
            model.eval()
            with torch.no_grad():
                output = call_model(model, input_tensor, self.use_amp)
                if self.is_cuda:
                    torch.cuda.synchronize()
                
                # Capture peak memory during inference
                peak_memory = torch.cuda.max_memory_allocated() / (1024 * 1024)
                current_memory = torch.cuda.memory_allocated() / (1024 * 1024)
                
                # Calculate memory component breakdown for optimization insights
                model_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 * 1024)
                input_memory = input_tensor.numel() * input_tensor.element_size() / (1024 * 1024)
                output_memory = output.numel() * output.element_size() / (1024 * 1024)
                activation_memory = peak_memory - baseline_memory - input_memory
            
            return {
                'baseline_memory_mb': baseline_memory,
                'peak_memory_mb': peak_memory,
                'current_memory_mb': current_memory,
                'memory_increase_mb': peak_memory - baseline_memory,
                'component_breakdown': {
                    'model_parameters_mb': model_memory,
                    'input_tensor_mb': input_memory,
                    'output_tensor_mb': output_memory,
                    'activations_mb': activation_memory
                }
            }
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                return {'error': 'GPU out of memory during profiling'}
            else:
                return {'error': f'GPU memory profiling failed: {str(e)}'}
    
    def profile_with_pytorch_profiler(self, model: nn.Module, input_tensor: torch.Tensor,
                                     num_steps: int = 10) -> Dict[str, Any]:
        """
        Enhanced PyTorch profiler for detailed operation-level performance analysis.
        
        Provides granular breakdown of computational operations, identifying
        performance bottlenecks and optimization opportunities. Essential for
        understanding where model optimization efforts should be focused.
        
        Args:
            model: PyTorch model to profile
            input_tensor: Representative input tensor
            num_steps: Number of profiling steps for statistical accuracy
            
        Returns:
            Dictionary containing:
                - operation_breakdown: Percentage breakdown by operation category
                - total_time_us: Total profiled time in microseconds
                - profiler_data: Raw PyTorch profiler object for detailed analysis
        
        Note:
            Operation categorization helps identify optimization targets:
            - Convolution operations: Often GPU-optimized, check kernel efficiency
            - Matrix multiply: Benefits from Tensor Cores on modern GPUs
            - Memory operations: May indicate data movement bottlenecks
            
        Example:
            >>> profiler = PerformanceProfiler()
            >>> results = profiler.profile_with_pytorch_profiler(model, input_tensor)
            >>> ops = results['operation_breakdown']
            >>> print(f"Convolution: {ops.get('convolution_ops', 0):.1f}%")
        """
        # Configure profiler activities based on available hardware
        activities = [torch.profiler.ProfilerActivity.CPU]
        if self.is_cuda:
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        
        model.eval()

        try:
            # Start comprehensive profiling with optimized settings
            with torch.profiler.profile(
                activities=activities,
                record_shapes=False,     # Disable to reduce overhead
                profile_memory=False,    # Keep simple to avoid overhead
                with_stack=False,        # Disable stack traces for performance
                with_flops=False         # Disable FLOP counting for cleaner results
            ) as prof:
                # Warmup phase - critical for stable GPU performance
                with torch.no_grad():
                    for _ in range(3):
                        _ = call_model(model, input_tensor, self.use_amp)
                        if self.is_cuda:
                            torch.cuda.synchronize()
                
                # Actual profiling measurements
                with torch.no_grad():
                    for _ in range(num_steps):
                        if self.is_cuda:
                            torch.cuda.synchronize()
                        _ = call_model(model, input_tensor, self.use_amp)
                        if self.is_cuda:
                            torch.cuda.synchronize()
            
            # Process operation breakdown for optimization insights
            operation_times = {}
            total_time = 0
            
            for event in prof.key_averages():
                # Select appropriate time measurement based on device
                if self.is_cuda and hasattr(event, 'cuda_time_total') and event.cuda_time_total > 0:
                    event_time = event.cuda_time_total
                else:
                    event_time = event.cpu_time_total
                
                if event_time <= 0:
                    continue
                    
                # Filter out profiling overhead operations
                event_name = event.key.lower()
                if any(skip in event_name for skip in ['memcpy', 'memset', 'sync', 'profiler']):
                    continue
                
                # Categorize operation for optimization targeting
                category = self._categorize_operation(event_name)
                operation_times[category] = operation_times.get(category, 0) + event_time
                total_time += event_time
            
            # Convert to percentages for relative analysis
            operation_percentages = {}
            if total_time > 0:
                operation_percentages = {op: (time_us / total_time) * 100 
                                    for op, time_us in operation_times.items()}
            
            return {
                'operation_breakdown': operation_percentages,
                'total_time_us': total_time,
                'profiler_data': prof
            }
            
        except Exception as e:
            return {'error': f'PyTorch profiler failed: {str(e)}'}


    def _categorize_operation(self, op_name: str) -> str:
        """
        Categorize PyTorch operations for optimization analysis.
        
        Groups low-level operations into high-level categories to identify
        optimization opportunities and performance bottlenecks.
        
        Args:
            op_name: PyTorch operation name from profiler
            
        Returns:
            Operation category string for grouping and analysis
        """
        if not op_name:
            return 'other_ops'
        
        op_name = op_name.lower()
    
        # Convolution operations - typically GPU-optimized
        if any(conv in op_name for conv in ['conv2d', 'convolution', '_convolution', 'mkldnn_convolution', 
                                           'depthwise', '_conv_depthwise2d', 'conv_depthwise2d_forward_kernel']):
            return 'convolution_ops'
        
        # Normalization operations - important for training stability
        elif any(norm in op_name for norm in ['batch_norm', 'layer_norm', 'group_norm', 'bn_fw', 'bn_bw', 'cudnn::bn']):
            return 'normalization_ops'
        
        # Pooling operations - spatial dimension reduction
        elif any(pool in op_name for pool in ['pool', 'mean', 'avg', 'adaptive']):
            return 'pooling_ops'
        
        # Matrix multiplication - benefits from Tensor Cores
        elif any(mm in op_name for mm in ['addmm', 'mm', 'bmm', 'linear', 'splitkreduce_kernel']):
            return 'matrix_multiply_ops'
        
        # Activation functions - element-wise operations
        elif any(act in op_name for act in ['relu', 'gelu', 'swish', 'silu', 'sigmoid', 'add_', 'mul_', 'mul', 
                                          'elementwise_kernel', 'cudafunctor_add', 'mulfunctor']):
            return 'activation_ops'
        
        # Memory operations - potential data movement bottlenecks
        elif any(mem in op_name for mem in ['copy', 'transpose', 'view', 'permute']):
            return 'memory_ops'
        
        else:
            return 'other_ops'

    def profile_multiple_batch_sizes(self, model: nn.Module, input_shape: Tuple[int, ...], 
                                    batch_sizes: List[int] = [1, 8, 16, 32, 64]) -> Dict[str, Any]:
        """
        Profile model performance across multiple batch sizes for optimal deployment configuration.
        
        Analyzes scaling behavior to identify optimal batch size for different deployment
        scenarios (real-time vs. batch processing). Critical for clinical applications
        where latency and throughput requirements vary.
        
        Args:
            model: PyTorch model to profile
            input_shape: Input tensor shape including batch dimension (batch, channels, height, width)
            batch_sizes: List of batch sizes to test for optimization analysis
            
        Returns:
            Dictionary with performance results for each batch size:
                - 'batch_N': Results for batch size N with timing and memory analysis
                - Includes error handling for out-of-memory conditions
        
        Note:
            Testing stops at first OOM error to prevent crashes. Results help
            determine optimal batch size for different deployment constraints.
            
        Example:
            >>> profiler = PerformanceProfiler()
            >>> results = profiler.profile_multiple_batch_sizes(
            ...     model, (1, 3, 224, 224), [1, 4, 8, 16]
            ... )
            >>> for batch_size, result in results.items():
            ...     if 'error' not in result:
            ...         timing = result['timing']
            ...         print(f"Batch {batch_size}: {timing['batch_efficiency']:.2f}x efficiency")
        """
        results = {}
        model.eval()
        
        for batch_size in batch_sizes:
            try:
                # Create input tensor for current batch size
                input_tensor = torch.randn(batch_size, *input_shape[1:]).to(self.device)
                
                # Profile timing performance
                timing_result = self.profile_inference_time(model, input_tensor, num_runs=30, warmup_runs=5)
                
                # Profile memory usage if CUDA available
                memory_result = {}
                if self.is_cuda:
                    memory_result = self.profile_memory_usage(model, input_tensor)
                
                results[f'batch_{batch_size}'] = {
                    'timing': timing_result,
                    'memory': memory_result
                }
                
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    results[f'batch_{batch_size}'] = {'error': 'GPU OOM'}
                    break  # Stop testing larger batch sizes
                else:
                    results[f'batch_{batch_size}'] = {'error': str(e)}
        
        return results


@contextmanager
def measure_time(operation_name: str) -> Iterator[None]:
    """
    Context manager for measuring and reporting operation execution time.
    
    Provides convenient timing measurement for code blocks with automatic
    reporting. Useful for quick performance checks during development.
    
    Args:
        operation_name: Descriptive name for the timed operation
        
    Yields:
        None: Context for the timed operation
        
    Example:
        >>> with measure_time("model inference"):
        ...     output = model(input_tensor)
        ⏱️ model inference took 15.23 ms
    """
    start_time = time.perf_counter()
    yield
    end_time = time.perf_counter()
    print(f"⏱️ {operation_name} took {(end_time - start_time) * 1000:.2f} ms")


def get_gpu_info() -> Dict[str, Any]:
    """
    Retrieve comprehensive GPU information for performance planning.
    
    Provides detailed hardware capabilities essential for optimization
    decisions and deployment planning in clinical environments.
    
    Returns:
        Dictionary containing:
            - Hardware specifications (name, memory, compute capability)
            - Feature support (Tensor Cores, mixed precision)
            - Performance characteristics for optimization planning
    
    Note:
        Tensor Core support (compute capability ≥ 7.0) enables significant
        speedups with mixed precision training and inference.
        
    Example:
        >>> gpu_info = get_gpu_info()
        >>> if gpu_info.get('tensor_core_support'):
        ...     print("Tensor Cores available - consider enabling AMP")
    """
    if not torch.cuda.is_available():
        return {'error': 'CUDA not available'}
    
    try:
        props = torch.cuda.get_device_properties(0)
        
        gpu_info = {
            'name': props.name,
            'memory_total_gb': props.total_memory / (1024**3),
            'memory_total_mb': props.total_memory / (1024**2),
            'compute_capability': f"{props.major}.{props.minor}",
            'multi_processor_count': props.multi_processor_count,
            'tensor_core_support': props.major >= 7,  # Volta and newer
            'mixed_precision_support': props.major >= 7  # AMP compatibility
        }
        
        return gpu_info
        
    except Exception as e:
        return {'error': f'Failed to get GPU info: {str(e)}'}


def check_environment() -> Dict[str, Any]:
    """
    Perform comprehensive environment compatibility check for profiling.
    
    Validates that the current environment supports all profiling features
    and provides recommendations for optimal configuration.
    
    Returns:
        Dictionary containing:
            - Software versions and compatibility flags
            - Hardware capabilities and availability
            - System resources and configuration recommendations
    
    Note:
        Ensures all required components are available before conducting
        performance analysis. Critical for reproducible results.
        
    Example:
        >>> env_info = check_environment()
        >>> if not env_info['pytorch_compatible']:
        ...     print("PyTorch version too old - upgrade recommended")
        >>> if env_info['cuda_available'] and env_info['profiler_support']:
        ...     print("Full GPU profiling available")
    """
    env_check = {
        'pytorch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'python_version': sys.version.split()[0],
        'platform': platform.platform()
    }
    
    # Check PyTorch version compatibility
    pytorch_major, pytorch_minor = map(int, torch.__version__.split('.')[:2])
    env_check['pytorch_compatible'] = pytorch_major >= 1 and pytorch_minor >= 9
    
    # Check GPU capabilities and CUDA environment
    if torch.cuda.is_available():
        env_check.update({
            'cuda_version': torch.version.cuda,
            'cudnn_available': torch.backends.cudnn.is_available(),
            'gpu_count': torch.cuda.device_count()
        })
        
        # Verify profiler support for detailed analysis
        try:
            torch.profiler.ProfilerActivity.CUDA
            env_check['profiler_support'] = True
        except:
            env_check['profiler_support'] = False
    
    # System resource information for batch size planning
    env_check.update({
        'system_memory_gb': psutil.virtual_memory().total / (1024**3),
        'cpu_count': psutil.cpu_count()
    })
    
    return env_check


def call_model(model: nn.Module, input_tensor: torch.Tensor, use_amp: bool = False) -> torch.Tensor:
    """
    Execute model inference with optional Automatic Mixed Precision (AMP).

    Provides a standardized interface for model inference with optional mixed precision
    optimization for CUDA devices. Ensures proper evaluation mode and gradient context.

    Args:
        model: PyTorch model to execute
        input_tensor: Input tensor for inference with shape (batch_size, channels, height, width)
        use_amp: Whether to enable AMP for mixed precision inference. Only effective
                when CUDA is available. Defaults to False.

    Returns:
        Model output tensor with shape (batch_size, num_classes)

    Note:
        AMP can significantly improve inference speed on modern GPUs with Tensor Cores
        while maintaining numerical stability for most models.
        
    Example:
        >>> model = create_baseline_model()
        >>> input_tensor = torch.randn(1, 3, 224, 224)
        >>> output = call_model(model, input_tensor, use_amp=True)
    """
    model.eval()
    with torch.no_grad():
        if use_amp and torch.cuda.is_available():
            # Use automatic mixed precision for faster inference on supported hardware
            with autocast():
                output = model(input_tensor)
        else:
            # Standard FP32 inference
            output = model(input_tensor)
    return output