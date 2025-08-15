"""
Comprehensive evaluation utilities for PneumoniaMNIST binary classification.

This module provides clinical-focused evaluation tools specifically designed for
binary pneumonia detection. Includes threshold optimization, comprehensive metrics
calculation, and clinical interpretation of results for medical imaging applications.

Key features:
    - Binary classification metrics with clinical context
    - Optimal threshold finding for sensitivity/specificity balance
    - Multiple threshold analysis for clinical decision support
    - Clinical interpretation with medical terminology
"""

from typing import Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
from sklearn.metrics import f1_score

import torch
import torch.nn.functional as F
import torch.utils.data
from torch import nn
from torchmetrics.classification import (
    BinaryAUROC,
    BinaryAccuracy,
    BinaryConfusionMatrix,
    BinaryF1Score,
    BinaryPrecision,
    BinaryRecall,
)


class ClassificationEvaluator:
    """
    Comprehensive evaluator for binary medical image classification.
    
    Provides standardized evaluation metrics specifically tailored for clinical
    applications where sensitivity (recall) and specificity are critical for
    patient safety and diagnostic accuracy.
    
    Attributes:
        device: Computing device for metric calculations
        threshold: Decision threshold for binary classification
        metrics: Dictionary of torchmetrics classification metrics
        confusion_matrix: Binary confusion matrix calculator
    """
    
    def __init__(self, device: str = 'cuda', threshold: float = 0.5) -> None:
        """
        Initialize evaluator with clinical-focused binary classification metrics.
        
        Args:
            device: Computing device ('cuda' or 'cpu') for metric calculations
            threshold: Decision threshold for converting probabilities to predictions.
                      0.5 is standard, but clinical applications may require
                      threshold optimization for sensitivity/specificity balance.
        
        Note:
            In medical imaging, threshold selection significantly impacts clinical
            outcomes. Lower thresholds increase sensitivity (fewer missed cases)
            but may increase false positives.
            
        Example:
            >>> evaluator = ClassificationEvaluator(device='cuda', threshold=0.4)
            >>> # Lower threshold prioritizes sensitivity for medical safety
        """
        self.device = device
        self.threshold = threshold
        
        # Initialize comprehensive binary classification metrics
        self.metrics = {
            'accuracy': BinaryAccuracy(threshold=threshold).to(self.device),
            'precision': BinaryPrecision(threshold=threshold).to(self.device),
            'recall': BinaryRecall(threshold=threshold).to(self.device),        # Sensitivity
            'f1': BinaryF1Score(threshold=threshold).to(self.device),
            'auc': BinaryAUROC().to(self.device),  # Threshold-independent metric
        }
        
        # Confusion matrix for detailed clinical analysis
        self.confusion_matrix = BinaryConfusionMatrix(threshold=threshold).to(self.device)

    def reset(self) -> None:
        """
        Reset all metrics for new evaluation session.
        
        Essential for evaluating multiple datasets or threshold configurations
        without metric contamination from previous evaluations.
        """
        for metric in self.metrics.values():
            metric.reset()
        self.confusion_matrix.reset()

    def update(self, outputs: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Update evaluation metrics with batch predictions.
        
        Processes model outputs and ground truth labels to update running
        metric calculations. Handles probability extraction from 2-class
        logits for proper binary classification evaluation.
        
        Args:
            outputs: Model logits with shape [batch_size, 2] for binary classification
            targets: Ground truth labels with shape [batch_size] containing 0 (normal) or 1 (pneumonia)
        
        Note:
            Converts 2-class logits to pneumonia probability for threshold-based
            classification. This is critical for medical applications where
            probability confidence affects clinical decision-making.
            
        Example:
            >>> evaluator = ClassificationEvaluator()
            >>> outputs = model(batch_images)  # Shape: [8, 2]
            >>> targets = batch_labels         # Shape: [8]
            >>> evaluator.update(outputs, targets)
        """
        # Convert 2-class logits to pneumonia probability (positive class)
        probs = F.softmax(outputs, dim=1)[:, 1]  # Probability of pneumonia detection
        
        # Ensure targets are properly formatted for binary classification
        targets = targets.view(-1).long()
        
        # Update all clinical metrics
        for metric in self.metrics.values():
            if metric == self.metrics['auc']:
                # AUC uses raw probabilities (threshold-independent)
                metric.update(probs, targets)
            else:
                # Other metrics use probabilities with threshold
                metric.update(probs, targets)
        
        # Update confusion matrix for clinical interpretation
        self.confusion_matrix.update(probs, targets)

    def compute(self) -> Dict[str, Union[float, np.ndarray]]:
        """
        Compute final evaluation metrics with clinical context.
        
        Returns:
            Dictionary containing:
                - Standard metrics: accuracy, precision, recall, f1, auc
                - Clinical metrics: sensitivity, specificity, PPV, NPV
                - Confusion matrix: for detailed diagnostic analysis
                - Threshold: used for probability-to-prediction conversion
        
        Note:
            Metrics are computed using accumulated predictions from all
            update() calls since last reset(). Essential for statistically
            significant evaluation on full datasets.
            
        Example:
            >>> results = evaluator.compute()
            >>> print(f"Sensitivity: {results['recall']:.1%}")  # Medical terminology
            >>> print(f"AUC-ROC: {results['auc']:.3f}")
        """
        results = {}
        
        # Compute all standard classification metrics
        for name, metric in self.metrics.items():
            value = metric.compute()
            results[name] = value.item() if hasattr(value, 'item') else float(value)
        
        # Add confusion matrix for detailed clinical analysis
        results['confusion_matrix'] = self.confusion_matrix.compute().cpu().numpy()
        
        # Add threshold metadata for clinical interpretation
        results['threshold'] = self.threshold
        
        return results


def find_optimal_threshold(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = 'cuda',
    metric: str = 'f1'
) -> float:
    """
    Find optimal decision threshold for clinical binary classification.
    
    Systematically evaluates multiple thresholds to identify the optimal
    operating point for clinical deployment. Critical for medical imaging
    where sensitivity/specificity balance affects patient outcomes.
    
    Args:
        model: Trained PyTorch model for pneumonia detection
        dataloader: Validation dataset for threshold optimization
        device: Computing device for model inference
        metric: Optimization target:
            - 'f1': Balanced F1 score (default)
            - 'balanced_accuracy': Equal weight to sensitivity and specificity
            - 'youden': Youden's J statistic (sensitivity + specificity - 1)
        
    Returns:
        Optimal threshold value for clinical deployment
    
    Note:
        Different metrics optimize for different clinical priorities:
        - F1: Balanced performance across both classes
        - Balanced accuracy: Equal importance to sensitivity and specificity
        - Youden: Maximizes true positive rate while minimizing false positive rate
        
    Example:
        >>> optimal_threshold = find_optimal_threshold(
        ...     model, val_loader, device='cuda', metric='balanced_accuracy'
        ... )
        >>> print(f"Optimal threshold for clinical deployment: {optimal_threshold:.3f}")
    """
    model.to(device)
    model.eval()
    
    # Collect all predictions and ground truth labels
    all_probs: List[float] = []
    all_targets: List[int] = []
    
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Get model predictions
            outputs = model(inputs)
            probs = F.softmax(outputs, dim=1)[:, 1]  # Pneumonia probability
            
            # Accumulate for threshold analysis
            all_probs.extend(probs.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    
    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)
    
    # Systematically evaluate threshold range
    thresholds = np.arange(0.1, 0.9, 0.01)  # Fine-grained search
    best_score = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        preds = (all_probs > threshold).astype(int)
        
        # Calculate metric based on clinical optimization target
        if metric == 'f1':
            score = f1_score(all_targets, preds)
        elif metric == 'balanced_accuracy':
            # Calculate sensitivity and specificity separately
            tp = np.sum((preds == 1) & (all_targets == 1))
            fn = np.sum((preds == 0) & (all_targets == 1))
            tn = np.sum((preds == 0) & (all_targets == 0))
            fp = np.sum((preds == 1) & (all_targets == 0))
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            score = (sensitivity + specificity) / 2
        elif metric == 'youden':
            # Youden's J statistic = sensitivity + specificity - 1
            tp = np.sum((preds == 1) & (all_targets == 1))
            fn = np.sum((preds == 0) & (all_targets == 1))
            tn = np.sum((preds == 0) & (all_targets == 0))
            fp = np.sum((preds == 1) & (all_targets == 0))
            
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            score = sensitivity + specificity - 1
        else:
            raise ValueError(f"Unsupported metric: {metric}")
        
        # Track best threshold for clinical deployment
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    print(f"Optimal threshold: {best_threshold:.3f} (optimizing {metric}: {best_score:.3f})")
    return best_threshold


def evaluate_model(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = 'cuda',
    threshold: float = 0.5,
    max_batches: Optional[int] = None
) -> Dict[str, Union[float, np.ndarray]]:
    """
    Comprehensive model evaluation on clinical dataset.
    
    Performs standardized evaluation with clinical metrics and detailed
    analysis suitable for medical imaging applications. Includes both
    threshold-dependent and threshold-independent metrics.
    
    Args:
        model: Trained PyTorch model for evaluation
        dataloader: Dataset for evaluation (validation or test set)
        device: Computing device for model inference
        threshold: Decision threshold for binary classification (default: 0.5)
        max_batches: Optional limit for quick evaluation during development
        
    Returns:
        Comprehensive evaluation dictionary containing:
            - Clinical metrics: sensitivity, specificity, PPV, NPV
            - Standard metrics: accuracy, precision, recall, F1, AUC
            - Confusion matrix: for detailed diagnostic analysis
    
    Note:
        Evaluation includes both threshold-dependent metrics (accuracy, precision)
        and threshold-independent metrics (AUC) for comprehensive clinical assessment.
        
    Example:
        >>> results = evaluate_model(model, test_loader, threshold=0.4)
        >>> print(format_evaluation_results(results))
    """
    model.to(device)
    model.eval()
    
    # Initialize evaluator with specified threshold
    evaluator = ClassificationEvaluator(device=device, threshold=threshold)
    evaluator.reset()
    
    # Process all batches for comprehensive evaluation
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(dataloader):
            if max_batches and i >= max_batches:
                break  # Early termination for development testing
                
            inputs = inputs.to(device)
            targets = targets.to(device)
            
            # Get model predictions and update metrics
            outputs = model(inputs)
            evaluator.update(outputs, targets)
    
    return evaluator.compute()


def format_evaluation_results(metrics: Dict[str, Union[float, np.ndarray]]) -> str:
    """
    Format evaluation results with clinical terminology and interpretation.
    
    Converts technical metrics into clinically meaningful format with
    medical terminology and practical interpretation for healthcare
    professionals and clinical deployment planning.
    
    Args:
        metrics: Dictionary of evaluation metrics from evaluate_model()
        
    Returns:
        Formatted string with clinical interpretation and technical metrics
    
    Note:
        Uses medical terminology (sensitivity, specificity, PPV) alongside
        technical metrics for comprehensive clinical reporting.
        
    Example:
        >>> results = evaluate_model(model, test_loader)
        >>> clinical_report = format_evaluation_results(results)
        >>> print(clinical_report)
    """
    cm = metrics['confusion_matrix']
    tn, fp, fn, tp = cm.ravel()
    
    # Calculate additional clinical metrics
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0  # Negative Predictive Value
    
    threshold_info = f" (threshold: {metrics.get('threshold', 0.5):.3f})" if 'threshold' in metrics else ""
    
    return (
        f"Clinical Evaluation Results{threshold_info}:\n"
        f"   Accuracy:  {metrics['accuracy']:.1%}\n"
        f"   Precision: {metrics['precision']:.1%} (PPV - Positive Predictive Value)\n"
        f"   Recall:    {metrics['recall']:.1%} (Sensitivity)\n"
        f"   F1 Score:  {metrics['f1']:.1%}\n"
        f"   AUC-ROC:   {metrics['auc']:.3f}\n"
        f"\nConfusion Matrix Analysis:\n"
        f"   True Negatives (Normal correctly identified):    {tn:4d}\n"
        f"   False Positives (Normal misclassified):          {fp:4d}\n"
        f"   False Negatives (Pneumonia missed):              {fn:4d}\n"
        f"   True Positives (Pneumonia correctly detected):  {tp:4d}\n"
        f"\nClinical Performance Interpretation:\n"
        f"   Sensitivity (True Positive Rate):     {metrics['recall']:.1%} "
        f"(detects {metrics['recall']*100:.1f}% of pneumonia cases)\n"
        f"   Specificity (True Negative Rate):     {specificity:.1%} "
        f"(correctly identifies {specificity*100:.1f}% of normal cases)\n"
        f"   PPV (Positive Predictive Value):      {metrics['precision']:.1%} "
        f"(when model predicts pneumonia, it's correct {metrics['precision']*100:.1f}% of time)\n"
        f"   NPV (Negative Predictive Value):      {npv:.1%} "
        f"(when model predicts normal, it's correct {npv*100:.1f}% of time)\n"
        f"\nClinical Significance:\n"
        f"   False Negative Rate: {fn/(tp+fn)*100:.1f}% (missed pneumonia cases - critical for patient safety)\n"
        f"   False Positive Rate: {fp/(fp+tn)*100:.1f}% (unnecessary follow-ups - impacts healthcare resources)"
    )


def evaluate_with_multiple_thresholds(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    device: str = 'cuda',
    thresholds: List[float] = [0.3, 0.4, 0.5, 0.6, 0.7]
) -> Dict[float, Dict[str, Union[float, np.ndarray]]]:
    """
    Evaluate model performance across multiple decision thresholds.
    
    Provides comprehensive threshold analysis to understand sensitivity/specificity
    trade-offs for clinical decision-making. Critical for medical imaging where
    different clinical scenarios may require different operating points.
    
    Args:
        model: Trained PyTorch model for evaluation
        dataloader: Dataset for multi-threshold evaluation
        device: Computing device for model inference
        thresholds: List of thresholds to evaluate for clinical comparison
        
    Returns:
        Dictionary mapping thresholds to evaluation results for comparison
    
    Note:
        Different thresholds optimize for different clinical priorities:
        - Lower thresholds: Higher sensitivity, fewer missed cases
        - Higher thresholds: Higher specificity, fewer false alarms
        Clinical deployment requires careful threshold selection based on use case.
        
    Example:
        >>> threshold_results = evaluate_with_multiple_thresholds(
        ...     model, test_loader, thresholds=[0.3, 0.5, 0.7]
        ... )
        >>> for threshold, results in threshold_results.items():
        ...     print(f"Threshold {threshold}: Sensitivity {results['recall']:.1%}")
    """
    results: Dict[float, Dict[str, Union[float, np.ndarray]]] = {}
    
    for threshold in thresholds:
        print(f"\nEvaluating with threshold {threshold:.1f}:")
        metrics = evaluate_model(model, dataloader, device, threshold)
        results[threshold] = metrics
        print(format_evaluation_results(metrics))
    
    return results