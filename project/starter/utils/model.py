"""
Model utilities for ResNet-18 baseline implementation and comprehensive model analysis.

This module provides model creation, architecture analysis, and training utilities
specifically designed for the PneumoniaMNIST binary classification project. Includes
adaptive input handling, transfer learning configuration, and clinical-focused training
procedures optimized for medical imaging applications.

Key features:
    - ResNet-18 baseline with adaptive input interpolation
    - Transfer learning from ImageNet with medical domain adaptation
    - Comprehensive model analysis and parameter counting
    - Clinical-focused training with early stopping and monitoring
"""

from typing import Any, Dict, Optional, Tuple, Union

import matplotlib.pyplot as plt
import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet18_Weights, resnet18


class ResNetBaseline(nn.Module):
    """
    ResNet-18 baseline model with adaptive input interpolation for medical imaging.
    
    This wrapper extends the standard ResNet-18 architecture to handle variable
    input sizes through automatic interpolation to ImageNet standard (224x224).
    Designed specifically for medical imaging transfer learning with clinical
    deployment considerations.
    
    Features:
        - Automatic input size adaptation for any resolution
        - ImageNet pretrained weights for transfer learning
        - Configurable fine-tuning for domain adaptation
        - Clinical-focused binary classification head
    
    Attributes:
        model: The underlying ResNet-18 architecture
        input_size: Expected input image size (metadata)
        target_size: Target size for interpolation (224x224)
        architecture_name: Human-readable architecture identifier
        num_classes: Number of output classes (2 for binary classification)
    """
    
    def __init__(self, num_classes: int = 2, input_size: int = 28, 
                 pretrained: bool = True, fine_tune: bool = False) -> None:
        """
        Initialize ResNet baseline model with medical imaging optimizations.
        
        Args:
            num_classes: Number of output classes (2 for normal vs pneumonia)
            input_size: Expected input image size for metadata tracking
            pretrained: Whether to use ImageNet pretrained weights for transfer learning
            fine_tune: Whether to enable fine-tuning of backbone layers
                      (recommended for medical domain adaptation)
        
        Note:
            Pretrained weights provide significant advantage for medical imaging
            as many low-level features (edges, textures) transfer effectively
            from natural images to medical images.
            
        Example:
            >>> model = ResNetBaseline(num_classes=2, pretrained=True, fine_tune=True)
            >>> print(f"Model architecture: {model.architecture_name}")
        """
        super().__init__()
        
        # Load ResNet-18 with optional pretrained weights
        if pretrained:
            self.model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        else:
            self.model = resnet18(weights=None)

        # Configure backbone fine-tuning for medical domain adaptation
        if not fine_tune:
            # Freeze backbone layers for feature extraction only
            for param in self.model.parameters():
                param.requires_grad = False

        # Build clinical classification head with regularization
        feature_dim = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Dropout(p=0.2),  # Regularization for medical imaging generalization
            nn.Linear(feature_dim, num_classes)
        )
        
        # Store model metadata for analysis and deployment
        self.input_size = input_size
        self.target_size = 224  # ImageNet standard for optimal pretrained performance
        self.architecture_name = "ResNet-18-Adaptive"
        self.num_classes = num_classes
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with adaptive input interpolation for variable input sizes.
        
        Automatically handles input size adaptation to leverage ImageNet pretrained
        features optimally. Critical for medical imaging where input sizes may vary
        across different acquisition protocols.
        
        Args:
            x: Input tensor of shape (batch_size, channels, height, width)
            
        Returns:
            Output logits of shape (batch_size, num_classes) for classification
            
        Note:
            Bilinear interpolation maintains spatial relationships in medical images
            while adapting to the expected ImageNet input size for optimal
            pretrained feature utilization.
            
        Example:
            >>> model = ResNetBaseline()
            >>> input_tensor = torch.randn(8, 3, 28, 28)  # Variable input size
            >>> output = model(input_tensor)  # Automatically interpolated to 224x224
            >>> print(f"Output shape: {output.shape}")  # [8, 2]
        """
        # Extract input dimensions for adaptive processing
        batch_size, channels, height, width = x.shape
        
        # Interpolate to ImageNet standard if input size differs
        if height != self.target_size or width != self.target_size:
            x = F.interpolate(
                x, 
                size=(self.target_size, self.target_size), 
                mode='bilinear',  # Preserves medical image spatial relationships
                align_corners=False
            )
        
        return self.model(x)


def create_baseline_model(num_classes: int = 2, input_size: int = 28, 
                         pretrained: bool = True, fine_tune: bool = True) -> ResNetBaseline:
    """
    Create ResNet-18 baseline model with clinical imaging optimizations.
    
    Factory function for creating standardized baseline models with medical
    imaging best practices. Provides consistent model configuration across
    experiments and deployment scenarios.
    
    Args:
        num_classes: Number of output classes (2 for binary pneumonia detection)
        input_size: Input image size (will be interpolated to 224x224 internally)
        pretrained: Whether to use ImageNet pretrained weights (recommended)
        fine_tune: Whether to enable backbone fine-tuning for domain adaptation
        
    Returns:
        Configured ResNetBaseline model ready for training or inference
        
    Note:
        Pretrained weights are strongly recommended for medical imaging as they
        provide robust feature extractors. Fine-tuning helps adapt to medical
        domain specifics while preserving general visual understanding.
        
    Example:
        >>> # Create model for pneumonia detection
        >>> model = create_baseline_model(
        ...     num_classes=2,
        ...     pretrained=True,
        ...     fine_tune=True
        ... )
        >>> print(f"Model ready for training on {model.input_size}x{model.input_size} images")
    """
    model = ResNetBaseline(
        num_classes=num_classes,
        input_size=input_size,
        pretrained=pretrained,
        fine_tune=fine_tune
    )
    
    return model


def get_model_info(model: nn.Module) -> Dict[str, Any]:
    """
    Extract comprehensive model information and architectural analysis.
    
    Provides detailed model characterization including parameter counts,
    memory requirements, and layer composition analysis. Essential for
    deployment planning and optimization targeting in clinical environments.
    
    Args:
        model: PyTorch model to analyze
        
    Returns:
        Dictionary containing:
            - Parameter counts (total and trainable)
            - Model size in MB for deployment planning
            - Layer composition breakdown for optimization
            - Architecture metadata for documentation
    
    Example:
        >>> model = create_baseline_model()
        >>> info = get_model_info(model)
        >>> print(f"Model: {info['architecture']}")
        >>> print(f"Parameters: {info['total_parameters']:,}")
        >>> print(f"Size: {info['model_size_mb']:.1f} MB")
    """
    # Calculate parameter statistics for deployment planning
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Calculate model size in MB (assuming FP32) for memory planning
    model_size_mb = (total_params * 4) / (1024 * 1024)
    
    # Analyze layer composition for optimization insights
    layer_breakdown = _analyze_layer_composition(model)
    
    return {
        'architecture': getattr(model, 'architecture_name', 'Unknown'),
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'model_size_mb': model_size_mb,
        'input_size': getattr(model, 'input_size', 'Unknown'),
        'layer_breakdown': layer_breakdown
    }


def _analyze_layer_composition(model: nn.Module) -> Dict[str, Any]:
    """
    Analyze model layer composition for optimization and deployment insights.
    
    Provides detailed breakdown of architectural components to guide
    optimization efforts and understand computational characteristics.
    
    Args:
        model: PyTorch model to analyze
        
    Returns:
        Dictionary with layer type counts, parameter distribution,
        and optimization insights for clinical deployment
    """
    composition = {
        'conv_layers': {'count': 0, 'total_params': 0},
        'linear_layers': {'count': 0, 'total_params': 0},
        'norm_layers': {'count': 0},
        'activation_layers': {'count': 0, 'types': set()},
        'other_layers': {'count': 0}
    }
    
    # Iterate through all model modules for comprehensive analysis
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Conv1d)):
            # Convolutional layers - primary computational load
            composition['conv_layers']['count'] += 1
            composition['conv_layers']['total_params'] += sum(p.numel() for p in module.parameters())
            
        elif isinstance(module, nn.Linear):
            # Linear layers - classifier and feature transformation
            composition['linear_layers']['count'] += 1
            composition['linear_layers']['total_params'] += sum(p.numel() for p in module.parameters())
            
        elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
            # Normalization layers - training stability and convergence
            composition['norm_layers']['count'] += 1
            
        elif isinstance(module, (nn.ReLU, nn.ReLU6, nn.SiLU, nn.Hardswish, nn.GELU)):
            # Activation functions - nonlinearity introduction
            composition['activation_layers']['count'] += 1
            composition['activation_layers']['types'].add(type(module).__name__)
            
        elif len(list(module.children())) == 0 and len(list(module.parameters())) > 0:
            # Other parameterized layers
            composition['other_layers']['count'] += 1
    
    # Convert set to list for JSON serialization compatibility
    composition['activation_layers']['types'] = list(composition['activation_layers']['types'])
    
    # Calculate parameter distribution for optimization insights
    total_params = composition['conv_layers']['total_params'] + composition['linear_layers']['total_params']
    if total_params > 0:
        composition['parameter_distribution'] = {
            'conv_percentage': (composition['conv_layers']['total_params'] / total_params) * 100,
            'linear_percentage': (composition['linear_layers']['total_params'] / total_params) * 100
        }
    
    return composition


def count_parameters_by_type(model: nn.Module) -> Dict[str, int]:
    """
    Count model parameters by layer type for detailed optimization analysis.
    
    Provides granular parameter breakdown to identify optimization targets
    and understand model complexity distribution.
    
    Args:
        model: PyTorch model to analyze
        
    Returns:
        Dictionary with parameter counts per layer type for optimization planning
        
    Example:
        >>> model = create_baseline_model()
        >>> param_counts = count_parameters_by_type(model)
        >>> print(f"Convolution parameters: {param_counts['conv2d']:,}")
        >>> print(f"Linear parameters: {param_counts['linear']:,}")
    """
    param_counts = {
        'conv2d': 0,      # Convolutional layers - feature extraction
        'linear': 0,      # Linear layers - classification
        'batchnorm': 0,   # Normalization layers - training stability
        'other': 0        # Miscellaneous layers
    }
    
    # Count parameters by layer type for optimization targeting
    for module in model.modules():
        params = sum(p.numel() for p in module.parameters(recurse=False))
        
        if isinstance(module, nn.Conv2d):
            param_counts['conv2d'] += params
        elif isinstance(module, nn.Linear):
            param_counts['linear'] += params
        elif isinstance(module, (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)):
            param_counts['batchnorm'] += params
        elif params > 0:
            param_counts['other'] += params
    
    return param_counts


def train_baseline_model(model: ResNetBaseline, train_loader, val_loader, device: str, 
                        config: Dict[str, Any], save_path: str = '../results/best_baseline_model.pth') -> Tuple[ResNetBaseline, Dict[str, list]]:
    """
    Train the baseline model with clinical-focused monitoring and optimization.
    
    Implements comprehensive training procedure with early stopping, learning rate
    scheduling, and clinical performance monitoring. Designed for medical imaging
    applications where model reliability and generalization are critical.
    
    Args:
        model: ResNetBaseline model to train
        train_loader: Training data loader with medical images
        val_loader: Validation data loader for performance monitoring
        device: Training device ('cuda' or 'cpu')
        config: Training configuration dictionary containing:
            - num_epochs: Maximum training epochs
            - learning_rate: Initial learning rate
            - weight_decay: L2 regularization strength
            - lr_step_size: Learning rate decay schedule
            - patience: Early stopping patience for clinical reliability
        save_path: Path to save best model for deployment
        
    Returns:
        Tuple containing:
            - Trained model with best validation performance
            - Training history dictionary with metrics for analysis
    
    Note:
        Early stopping based on validation accuracy ensures clinical reliability
        and prevents overfitting to training data. Essential for medical imaging
        where generalization to unseen cases is critical.
        
    Example:
        >>> model = create_baseline_model()
        >>> config = {
        ...     'num_epochs': 50,
        ...     'learning_rate': 0.001,
        ...     'weight_decay': 0.01,
        ...     'patience': 10
        ... }
        >>> trained_model, history = train_baseline_model(
        ...     model, train_loader, val_loader, 'cuda', config
        ... )
    """
    print(f"Starting baseline model training for pneumonia detection...")
    print(f"   Config: {config['num_epochs']} epochs, lr={config['learning_rate']}, wd={config['weight_decay']}")
    
    # Setup training components optimized for medical imaging
    criterion = nn.CrossEntropyLoss()  # Standard for binary classification
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']  # L2 regularization for generalization
    )

    # Learning rate scheduler for training stability
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=config['lr_step_size'], 
        gamma=0.1
    )

    # Training history tracking for analysis and monitoring
    history = {
        'train_loss': [], 'val_loss': [],
        'train_acc': [], 'val_acc': [], 'lr': []
    }
    
    # Early stopping for clinical reliability
    best_val_acc = 0.0
    patience_counter = 0
    
    # Main training loop with comprehensive monitoring
    for epoch in range(config['num_epochs']):
        # Training phase with gradient updates
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        # Progress tracking for training monitoring
        train_pbar = tqdm.tqdm(train_loader, desc=f'Epoch {epoch+1}/{config["num_epochs"]} [Train]', 
                              leave=False, ncols=100)
        
        for batch_idx, (data, target) in enumerate(train_pbar):
            # Move data to training device
            data = data.to(device, memory_format=config.get('memory_format', torch.preserve_format))
            target = target.to(device)

            # Convert binary labels to class indices for CrossEntropyLoss
            if target.dim() > 1 and target.size(1) == 1:
                target = target.squeeze(1).long()
            
            # Forward pass and loss calculation
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            # Gradient clipping for training stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()

            # Training statistics tracking
            train_loss += loss.item()
            pred = output.argmax(dim=1)
            train_correct += pred.eq(target).sum().item()
            train_total += target.size(0)
            
            # Real-time progress monitoring
            current_acc = 100. * train_correct / train_total
            train_pbar.set_postfix({'Loss': f'{loss.item():.3f}', 'Acc': f'{current_acc:.1f}%'})
        
        # Validation phase for performance monitoring
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        
        val_pbar = tqdm.tqdm(val_loader, desc=f'Epoch {epoch+1}/{config["num_epochs"]} [Val]', 
                            leave=False, ncols=100)
        
        with torch.no_grad():
            for data, target in val_pbar:
                data, target = data.to(device), target.to(device)
                
                if target.dim() > 1 and target.size(1) == 1:
                    target = target.squeeze(1).long()
                
                output = model(data)
                val_loss += criterion(output, target).item()
                pred = output.argmax(dim=1)
                val_correct += pred.eq(target).sum().item()
                val_total += target.size(0)
                
                # Real-time validation monitoring
                current_acc = 100. * val_correct / val_total
                val_pbar.set_postfix({'Loss': f'{val_loss/(val_pbar.n+1):.3f}', 'Acc': f'{current_acc:.1f}%'})
            
        # Calculate epoch performance metrics
        train_acc = 100. * train_correct / train_total
        val_acc = 100. * val_correct / val_total
        train_loss /= len(train_loader)
        val_loss /= len(val_loader)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Update training history for analysis
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)
        history['lr'].append(current_lr)
        
        # Learning rate scheduling
        scheduler.step()

        # Comprehensive epoch summary
        print(f"   Epoch {epoch+1:2d}: Train Acc {train_acc:5.1f}% | "
              f"Val Acc {val_acc:5.1f}% | Train Loss {train_loss:.4f} | "
              f"Val Loss {val_loss:.4f} | LR {current_lr:.6f}")
        
        # Early stopping with validation improvement tracking
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save best model for clinical deployment
            torch.save(model.state_dict(), save_path)
            print(f"      New best model saved (Val Acc: {val_acc:.1f}%)")
        else:
            patience_counter += 1
            if patience_counter >= config['patience']:
                print(f"     Early stopping triggered after {epoch+1} epochs "
                      f"(patience: {config['patience']})")
                break
    
    # Load best model for deployment
    model.load_state_dict(torch.load(save_path))
    
    print(f"Training completed! Best validation accuracy: {best_val_acc:.2f}%")
    return model, history


def plot_training_history(history: Dict[str, list]) -> None:
    """
    Visualize training progress with loss and accuracy curves.
    
    Creates comprehensive training visualization for analysis and reporting.
    Essential for understanding model convergence and identifying potential
    training issues in medical imaging applications.
    
    Args:
        history: Training history dictionary from train_baseline_model()
        
    Example:
        >>> model, history = train_baseline_model(model, train_loader, val_loader, device, config)
        >>> plot_training_history(history)
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss progression analysis
    ax1.plot(epochs, history['train_loss'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_title('Training and Validation Loss', fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy progression analysis
    ax2.plot(epochs, history['train_acc'], 'b-', label='Training Accuracy', linewidth=2)
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation Accuracy', linewidth=2)
    ax2.set_title('Training and Validation Accuracy', fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Training summary for clinical reporting
    print(f"\nTraining Summary:")
    print(f"   Best Validation Accuracy: {max(history['val_acc']):.1f}%")
    print(f"   Total Epochs: {len(epochs)}")
    print(f"   Final Training Accuracy: {history['train_acc'][-1]:.1f}%")
    print(f"   Final Validation Accuracy: {history['val_acc'][-1]:.1f}%")