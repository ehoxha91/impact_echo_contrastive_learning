import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.insert(0, '/Users/evhoxha/projects/impact_echo_contrastive_learning/')
sys.path.insert(0, 'dataloaders/')
sys.path.insert(0, 'models/')
sys.path.insert(0, 'data/')
sys.path.insert(0, 'weights/')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataloaders.dataloader import ImpactEchoDatasetClassifier
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from scipy import stats
from sklearn.metrics import confusion_matrix, classification_report, precision_recall_fscore_support
from utils import *

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

from models.model_parts import ResidualBlock
from train_evidential_multiclass import MulticlassEvidentialIENet, MulticlassImpactEchoDataset

# Default configuration
default_experiment_name = "evidential_multiclass_v1"
default_model_name = "evidential_multiclass_v1"


def calculate_multiclass_accuracy_metrics(pred_classes, targets, num_classes=5, class_names=None):
    """
    Calculate comprehensive multi-class accuracy evaluation metrics
    """
    if class_names is None:
        class_names = ['Solid Concrete', 'Defect Type 1', 'Defect Type 2', 'Defect Type 3', 'Defect Type 4']
    
    # Convert to numpy if tensors
    if hasattr(pred_classes, 'cpu'):
        pred_classes = pred_classes.cpu().numpy()
    if hasattr(targets, 'cpu'):
        targets = targets.cpu().numpy()
    
    # Ensure arrays are 1D
    pred_classes = pred_classes.flatten()
    targets = targets.flatten()
    
    # Remove any samples with invalid targets
    valid_mask = (targets >= 0) & (targets < num_classes)
    pred_classes = pred_classes[valid_mask]
    targets = targets[valid_mask]
    
    if len(targets) == 0:
        print("Warning: No valid targets found for accuracy calculation")
        return None
    
    print(f"\n=== Multi-class Defect Classification Metrics ({num_classes} classes) ===")
    print(f"Total samples evaluated: {len(targets)}")
    
    # Class distribution
    print(f"\nClass distribution:")
    for i in range(num_classes):
        count = np.sum(targets == i)
        print(f"  {class_names[i]}: {count} samples ({count/len(targets)*100:.1f}%)")
    
    # Confusion Matrix
    cm = confusion_matrix(targets, pred_classes, labels=range(num_classes))
    print(f"\nConfusion Matrix ({num_classes}x{num_classes}):")
    print("      Predicted")
    print("     ", end="")
    for i in range(num_classes):
        print(f"{i:>6}", end="")
    print()
    for i in range(num_classes):
        print(f"True {i:>2}:", end="")
        for j in range(num_classes):
            print(f"{cm[i,j]:>6}", end="")
        print()
    
    # Overall accuracy
    overall_accuracy = np.sum(pred_classes == targets) / len(targets)
    print(f"\nOverall Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.2f}%)")
    
    # Per-class metrics
    print(f"\n=== Per-Class Performance ===")
    for class_idx in range(num_classes):
        class_mask = targets == class_idx
        if class_mask.sum() > 0:
            class_accuracy = np.sum((targets == class_idx) & (pred_classes == class_idx)) / class_mask.sum()
            precision = cm[class_idx, class_idx] / max(1, np.sum(cm[:, class_idx]))
            recall = cm[class_idx, class_idx] / max(1, np.sum(cm[class_idx, :]))
            f1 = 2 * precision * recall / max(1e-8, precision + recall)
            
            print(f"{class_names[class_idx]}:")
            print(f"  Accuracy: {class_accuracy:.4f} ({class_accuracy*100:.2f}%)")
            print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
            print(f"  Recall: {recall:.4f} ({recall*100:.2f}%)")
            print(f"  F1-Score: {f1:.4f}")
    
    # Macro and micro averages
    try:
        precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
            targets, pred_classes, average='macro', zero_division=0, labels=range(num_classes)
        )
        precision_micro, recall_micro, f1_micro, _ = precision_recall_fscore_support(
            targets, pred_classes, average='micro', zero_division=0, labels=range(num_classes)
        )
        
        print(f"\n=== Average Metrics ===")
        print(f"Macro Average - Precision: {precision_macro:.4f}, Recall: {recall_macro:.4f}, F1: {f1_macro:.4f}")
        print(f"Micro Average - Precision: {precision_micro:.4f}, Recall: {recall_micro:.4f}, F1: {f1_micro:.4f}")
        
        # Detailed classification report
        print(f"\n=== Detailed Classification Report ===")
        report = classification_report(targets, pred_classes, target_names=class_names, zero_division=0, labels=range(num_classes))
        print(report)
        
    except Exception as e:
        print(f"Warning: Could not generate detailed metrics: {e}")
    
    # Return metrics dictionary
    metrics = {
        'overall_accuracy': overall_accuracy,
        'confusion_matrix': cm,
        'class_counts': [np.sum(targets == i) for i in range(num_classes)],
        'num_classes': num_classes,
        'class_names': class_names
    }
    
    return metrics


def analyze_multiclass_evidential_results(results_path, dataset_name="Test"):
    """
    Analyze and visualize multi-class evidential deep learning results
    """
    print(f"Loading multi-class results from {results_path}")
    results = torch.load(results_path, map_location='cpu')
    
    predictions = results['predictions']
    total_unc = results['total_uncertainties']
    epistemic_unc = results['epistemic_uncertainties']
    aleatoric_unc = results['aleatoric_uncertainties']
    confidences = results['confidences']
    alphas = results['alphas']
    targets = results['targets']
    accuracy = results['accuracy']
    num_classes = results['model_config']['num_classes']
    
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Number of test samples: {len(targets)}")
    print(f"Number of classes: {num_classes}")
    
    # Extract predictions for analysis
    pred_probs = predictions.squeeze(0)  # Remove sequence dim
    pred_classes = torch.argmax(pred_probs, dim=1)
    
    # Remove extra dimensions
    epistemic_unc = epistemic_unc.squeeze(0).squeeze(1)
    aleatoric_unc = aleatoric_unc.squeeze(0).squeeze(1)
    total_unc = total_unc.squeeze(0).squeeze(1)
    alphas = alphas.squeeze(0).squeeze(1)
    
    # Convert to numpy for plotting
    epistemic_unc_np = epistemic_unc.detach().cpu().numpy()
    aleatoric_unc_np = aleatoric_unc.detach().cpu().numpy()
    total_unc_np = total_unc.detach().cpu().numpy()
    alphas_np = alphas.detach().cpu().numpy()
    confidences_np = confidences.detach().cpu().numpy()
    pred_classes_np = pred_classes.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()
    pred_probs_np = pred_probs.detach().cpu().numpy()
    
    # Multi-class classification metrics
    correct_predictions = (pred_classes_np == targets_np)
    incorrect_predictions = ~correct_predictions
    
    print(f"\nCorrect Predictions: {correct_predictions.sum()}/{len(targets_np)}")
    print(f"Incorrect Predictions: {incorrect_predictions.sum()}/{len(targets_np)}")
    
    # Calculate detailed multi-class accuracy metrics
    detailed_metrics = calculate_multiclass_accuracy_metrics(pred_classes_np, targets_np, num_classes)
    
    # Comprehensive uncertainty statistics
    print(f"\n=== Multi-class Uncertainty Analysis ===")
    print(f"Total Uncertainty - Mean: {np.mean(total_unc_np):.6f} ± {np.std(total_unc_np):.6f}")
    print(f"Epistemic Uncertainty - Mean: {np.mean(epistemic_unc_np):.6f} ± {np.std(epistemic_unc_np):.6f}")
    print(f"Aleatoric Uncertainty - Mean: {np.mean(aleatoric_unc_np):.6f} ± {np.std(aleatoric_unc_np):.6f}")
    print(f"Confidence - Mean: {np.mean(confidences_np):.6f} ± {np.std(confidences_np):.6f}")
    print(f"Evidence Strength (Alpha Sum) - Mean: {np.mean(alphas_np):.6f} ± {np.std(alphas_np):.6f}")
    
    # Uncertainty for correct vs incorrect predictions
    print(f"\n=== Correctness-based Analysis ===")
    if correct_predictions.sum() > 0:
        print(f"Correct Predictions:")
        print(f"  - Mean Total Uncertainty: {np.mean(total_unc_np[correct_predictions]):.6f}")
        print(f"  - Mean Epistemic Uncertainty: {np.mean(epistemic_unc_np[correct_predictions]):.6f}")
        print(f"  - Mean Aleatoric Uncertainty: {np.mean(aleatoric_unc_np[correct_predictions]):.6f}")
        print(f"  - Mean Confidence: {np.mean(confidences_np[correct_predictions]):.6f}")
        print(f"  - Mean Evidence Strength: {np.mean(alphas_np[correct_predictions]):.6f}")
    
    if incorrect_predictions.sum() > 0:
        print(f"Incorrect Predictions:")
        print(f"  - Mean Total Uncertainty: {np.mean(total_unc_np[incorrect_predictions]):.6f}")
        print(f"  - Mean Epistemic Uncertainty: {np.mean(epistemic_unc_np[incorrect_predictions]):.6f}")
        print(f"  - Mean Aleatoric Uncertainty: {np.mean(aleatoric_unc_np[incorrect_predictions]):.6f}")
        print(f"  - Mean Confidence: {np.mean(confidences_np[incorrect_predictions]):.6f}")
        print(f"  - Mean Evidence Strength: {np.mean(alphas_np[incorrect_predictions]):.6f}")
    
    # Class-specific analysis
    print(f"\n=== Class-specific Uncertainty Analysis ===")
    class_names = ['Solid Concrete', 'Defect Type 1', 'Defect Type 2', 'Defect Type 3', 'Defect Type 4']
    for class_idx in range(num_classes):
        class_mask = targets_np == class_idx
        if class_mask.sum() > 0:
            class_name = class_names[class_idx] if class_idx < len(class_names) else f"Class {class_idx}"
            class_accuracy = (pred_classes_np[class_mask] == class_idx).mean()
            print(f"{class_name} (Class {class_idx}) - {class_mask.sum()} samples:")
            print(f"  - Class Accuracy: {class_accuracy*100:.2f}%")
            print(f"  - Mean Total Uncertainty: {np.mean(total_unc_np[class_mask]):.6f}")
            print(f"  - Mean Epistemic Uncertainty: {np.mean(epistemic_unc_np[class_mask]):.6f}")
            print(f"  - Mean Confidence: {np.mean(confidences_np[class_mask]):.6f}")
    
    # Create visualizations
    print("\n=== Creating Multi-class Uncertainty Visualizations ===")
    
    # 1. Multi-class uncertainty distribution analysis
    create_multiclass_uncertainty_distributions(
        pred_probs_np, epistemic_unc_np, aleatoric_unc_np, total_unc_np, 
        confidences_np, alphas_np, targets_np, correct_predictions, num_classes
    )
    
    # 2. Multi-class confusion matrix heatmap
    if detailed_metrics:
        create_multiclass_confusion_matrix_plot(detailed_metrics['confusion_matrix'], class_names[:num_classes])
    
    # 3. Class-wise uncertainty comparison
    create_classwise_uncertainty_comparison(
        epistemic_unc_np, aleatoric_unc_np, total_unc_np, 
        confidences_np, targets_np, num_classes, class_names[:num_classes]
    )
    
    # 4. Multi-class prediction confidence analysis
    create_multiclass_confidence_analysis(pred_probs_np, targets_np, num_classes, class_names[:num_classes])
    
    return results


def create_multiclass_uncertainty_distributions(pred_probs, epistemic_unc, aleatoric_unc, total_unc, 
                                              confidences, alphas, targets, correct_predictions, num_classes,
                                              experiment_name=None, model_name=None):
    """
    Create multi-class uncertainty distribution analysis plots
    """
    if experiment_name is None:
        experiment_name = default_experiment_name
    if model_name is None:
        model_name = default_model_name
    
    plt.style.use('default')
    sns.set_palette("husl")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Multi-class Evidential Uncertainty Analysis ({num_classes} classes)', fontsize=18, fontweight='bold')
    
    # Plot 1: Epistemic vs Aleatoric Uncertainty scatter
    axes[0, 0].scatter(epistemic_unc[correct_predictions], aleatoric_unc[correct_predictions], 
                      alpha=0.7, c='green', label='Correct', s=40, edgecolors='black', linewidth=0.5)
    axes[0, 0].scatter(epistemic_unc[~correct_predictions], aleatoric_unc[~correct_predictions], 
                      alpha=0.7, c='red', label='Incorrect', s=40, edgecolors='black', linewidth=0.5)
    axes[0, 0].set_xlabel('Epistemic Uncertainty', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('Aleatoric Uncertainty', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('Epistemic vs Aleatoric Uncertainty', fontsize=14, fontweight='bold')
    axes[0, 0].legend(frameon=True, fancybox=True, shadow=True)
    axes[0, 0].grid(True, alpha=0.3, linestyle='--')
    
    # Plot 2: Total Uncertainty Distribution histogram
    axes[0, 1].hist(total_unc[correct_predictions], bins=25, alpha=0.7, 
                   color='green', label='Correct', density=True, edgecolor='black', linewidth=0.8)
    axes[0, 1].hist(total_unc[~correct_predictions], bins=25, alpha=0.7, 
                   color='red', label='Incorrect', density=True, edgecolor='black', linewidth=0.8)
    axes[0, 1].set_xlabel('Total Uncertainty', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Density', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Total Uncertainty Distribution', fontsize=14, fontweight='bold')
    axes[0, 1].legend(frameon=True, fancybox=True, shadow=True)
    axes[0, 1].grid(True, alpha=0.3, linestyle='--')
    
    # Plot 3: Confidence Distribution histogram
    axes[0, 2].hist(confidences[correct_predictions], bins=25, alpha=0.7, 
                   color='green', label='Correct', density=True, edgecolor='black', linewidth=0.8)
    axes[0, 2].hist(confidences[~correct_predictions], bins=25, alpha=0.7, 
                   color='red', label='Incorrect', density=True, edgecolor='black', linewidth=0.8)
    axes[0, 2].set_xlabel('Confidence', fontsize=12, fontweight='bold')
    axes[0, 2].set_ylabel('Density', fontsize=12, fontweight='bold')
    axes[0, 2].set_title('Confidence Distribution', fontsize=14, fontweight='bold')
    axes[0, 2].legend(frameon=True, fancybox=True, shadow=True)
    axes[0, 2].grid(True, alpha=0.3, linestyle='--')
    
    # Plot 4: Class-wise uncertainty boxplot
    class_uncertainties = [total_unc[targets == i] for i in range(num_classes) if np.sum(targets == i) > 0]
    class_labels = [f'Class {i}' for i in range(num_classes) if np.sum(targets == i) > 0]
    
    if class_uncertainties:
        axes[1, 0].boxplot(class_uncertainties, labels=class_labels)
        axes[1, 0].set_ylabel('Total Uncertainty', fontsize=12, fontweight='bold')
        axes[1, 0].set_title('Uncertainty by True Class', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3, linestyle='--')
        axes[1, 0].tick_params(axis='x', rotation=45)
    
    # Plot 5: Evidence strength vs confidence
    scatter = axes[1, 1].scatter(alphas, confidences, c=correct_predictions, 
                               cmap='RdYlGn', alpha=0.6, s=40, edgecolors='black', linewidth=0.5)
    axes[1, 1].set_xlabel('Evidence Strength (Alpha Sum)', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Confidence', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('Evidence Strength vs Confidence', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, linestyle='--')
    cbar = plt.colorbar(scatter, ax=axes[1, 1])
    cbar.set_label('Correct Prediction', fontsize=10)
    
    # Plot 6: Uncertainty vs Confidence scatter (colored by class)
    scatter2 = axes[1, 2].scatter(total_unc, confidences, c=targets, 
                                cmap='tab10', alpha=0.6, s=40, edgecolors='black', linewidth=0.5)
    axes[1, 2].set_xlabel('Total Uncertainty', fontsize=12, fontweight='bold')
    axes[1, 2].set_ylabel('Confidence', fontsize=12, fontweight='bold')
    axes[1, 2].set_title('Uncertainty vs Confidence (by Class)', fontsize=14, fontweight='bold')
    axes[1, 2].grid(True, alpha=0.3, linestyle='--')
    cbar2 = plt.colorbar(scatter2, ax=axes[1, 2])
    cbar2.set_label('True Class', fontsize=10)
    
    plt.tight_layout()
    
    # Create experiment directory
    import os
    os.makedirs(f'new_uncertainty_results/{experiment_name}', exist_ok=True)
    
    plt.savefig(f'new_uncertainty_results/{experiment_name}/{model_name}_multiclass_uncertainty_distributions.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Multi-class uncertainty distribution analysis saved")


def create_multiclass_confusion_matrix_plot(confusion_matrix, class_names, experiment_name=None, model_name=None):
    """
    Create a beautiful confusion matrix heatmap for multi-class classification
    """
    if experiment_name is None:
        experiment_name = default_experiment_name
    if model_name is None:
        model_name = default_model_name
    
    plt.figure(figsize=(10, 8))
    
    # Normalize confusion matrix
    cm_normalized = confusion_matrix.astype('float') / confusion_matrix.sum(axis=1)[:, np.newaxis]
    
    # Create heatmap
    sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                square=True, cbar_kws={"shrink": .8})
    
    plt.title('Multi-class Confusion Matrix (Normalized)', fontsize=16, fontweight='bold')
    plt.xlabel('Predicted Class', fontsize=14, fontweight='bold')
    plt.ylabel('True Class', fontsize=14, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    plt.savefig(f'new_uncertainty_results/{experiment_name}/{model_name}_multiclass_confusion_matrix.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Multi-class confusion matrix saved")


def create_classwise_uncertainty_comparison(epistemic_unc, aleatoric_unc, total_unc, 
                                          confidences, targets, num_classes, class_names,
                                          experiment_name=None, model_name=None):
    """
    Create class-wise uncertainty comparison plots
    """
    if experiment_name is None:
        experiment_name = default_experiment_name
    if model_name is None:
        model_name = default_model_name
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Class-wise Uncertainty Comparison', fontsize=16, fontweight='bold')
    
    # Prepare data for each class
    class_data = {}
    for class_idx in range(num_classes):
        class_mask = targets == class_idx
        if class_mask.sum() > 0:
            class_data[class_idx] = {
                'epistemic': epistemic_unc[class_mask],
                'aleatoric': aleatoric_unc[class_mask],
                'total': total_unc[class_mask],
                'confidence': confidences[class_mask]
            }
    
    # Plot 1: Epistemic uncertainty by class
    epistemic_data = [class_data[i]['epistemic'] for i in class_data.keys()]
    epistemic_labels = [class_names[i] for i in class_data.keys()]
    
    axes[0, 0].boxplot(epistemic_data, labels=epistemic_labels)
    axes[0, 0].set_ylabel('Epistemic Uncertainty')
    axes[0, 0].set_title('Epistemic Uncertainty by Class')
    axes[0, 0].tick_params(axis='x', rotation=45)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Aleatoric uncertainty by class
    aleatoric_data = [class_data[i]['aleatoric'] for i in class_data.keys()]
    
    axes[0, 1].boxplot(aleatoric_data, labels=epistemic_labels)
    axes[0, 1].set_ylabel('Aleatoric Uncertainty')
    axes[0, 1].set_title('Aleatoric Uncertainty by Class')
    axes[0, 1].tick_params(axis='x', rotation=45)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Total uncertainty by class
    total_data = [class_data[i]['total'] for i in class_data.keys()]
    
    axes[1, 0].boxplot(total_data, labels=epistemic_labels)
    axes[1, 0].set_ylabel('Total Uncertainty')
    axes[1, 0].set_title('Total Uncertainty by Class')
    axes[1, 0].tick_params(axis='x', rotation=45)
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Confidence by class
    confidence_data = [class_data[i]['confidence'] for i in class_data.keys()]
    
    axes[1, 1].boxplot(confidence_data, labels=epistemic_labels)
    axes[1, 1].set_ylabel('Confidence')
    axes[1, 1].set_title('Confidence by Class')
    axes[1, 1].tick_params(axis='x', rotation=45)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'new_uncertainty_results/{experiment_name}/{model_name}_classwise_uncertainty_comparison.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Class-wise uncertainty comparison saved")


def create_multiclass_confidence_analysis(pred_probs, targets, num_classes, class_names,
                                         experiment_name=None, model_name=None):
    """
    Create multi-class prediction confidence analysis
    """
    if experiment_name is None:
        experiment_name = default_experiment_name
    if model_name is None:
        model_name = default_model_name
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Multi-class Prediction Confidence Analysis', fontsize=16, fontweight='bold')
    
    # Get prediction confidences (max probability)
    max_probs = np.max(pred_probs, axis=1)
    pred_classes = np.argmax(pred_probs, axis=1)
    
    # Plot 1: Confidence distribution by correctness
    correct_mask = pred_classes == targets
    
    axes[0, 0].hist(max_probs[correct_mask], bins=25, alpha=0.7, color='green', 
                   label='Correct', density=True, edgecolor='black')
    axes[0, 0].hist(max_probs[~correct_mask], bins=25, alpha=0.7, color='red', 
                   label='Incorrect', density=True, edgecolor='black')
    axes[0, 0].set_xlabel('Prediction Confidence')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Confidence Distribution by Correctness')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Confidence by true class
    class_confidences = []
    class_labels_present = []
    for class_idx in range(num_classes):
        class_mask = targets == class_idx
        if class_mask.sum() > 0:
            class_confidences.append(max_probs[class_mask])
            class_labels_present.append(class_names[class_idx])
    
    if class_confidences:
        axes[0, 1].boxplot(class_confidences, labels=class_labels_present)
        axes[0, 1].set_ylabel('Prediction Confidence')
        axes[0, 1].set_title('Confidence by True Class')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Per-class probability distributions
    n_classes_to_plot = min(num_classes, 5)  # Limit to 5 classes for readability
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes_to_plot))
    
    for i in range(n_classes_to_plot):
        if np.sum(targets == i) > 0:
            class_probs = pred_probs[targets == i, i]  # Probability of correct class for samples of class i
            axes[1, 0].hist(class_probs, bins=20, alpha=0.7, color=colors[i], 
                           label=f'{class_names[i]}', density=True, edgecolor='black')
    
    axes[1, 0].set_xlabel('Probability of True Class')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('True Class Probability Distributions')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 4: Calibration plot (reliability diagram)
    # Bin predictions by confidence and check accuracy in each bin
    n_bins = 10
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    accuracies = []
    confidences = []
    bin_counts = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (max_probs > bin_lower) & (max_probs <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = correct_mask[in_bin].mean()
            avg_confidence_in_bin = max_probs[in_bin].mean()
            
            accuracies.append(accuracy_in_bin)
            confidences.append(avg_confidence_in_bin)
            bin_counts.append(in_bin.sum())
        else:
            accuracies.append(0)
            confidences.append((bin_lower + bin_upper) / 2)
            bin_counts.append(0)
    
    axes[1, 1].bar(range(len(accuracies)), accuracies, alpha=0.7, color='skyblue', edgecolor='black')
    axes[1, 1].plot([0, len(accuracies)], [0, 1], 'r--', label='Perfect Calibration')
    axes[1, 1].set_xlabel('Confidence Bin')
    axes[1, 1].set_ylabel('Accuracy')
    axes[1, 1].set_title('Calibration Plot')
    axes[1, 1].set_xticks(range(len(bin_lowers)))
    axes[1, 1].set_xticklabels([f'{bl:.1f}-{bu:.1f}' for bl, bu in zip(bin_lowers, bin_uppers)], rotation=45)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'new_uncertainty_results/{experiment_name}/{model_name}_multiclass_confidence_analysis.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Multi-class confidence analysis saved")


def test_multiclass_evidential_model_on_datasets(model_path):
    """
    Test the trained multi-class evidential model on all available datasets and generate maps
    """
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Load model
    model = MulticlassEvidentialIENet(num_classes=5, verbose=False).to(device)
    
    try:
        # Try loading as a full checkpoint first (with training metadata)
        checkpoint = torch.load(model_path, map_location=device)
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded multi-class evidential model from checkpoint: {model_path}")
        else:
            # Try loading as direct state dict
            model.load_state_dict(checkpoint)
            print(f"Loaded multi-class evidential model from state dict: {model_path}")
    except RuntimeError as e:
        print(f"Error loading model: {e}")
        print("This might be due to model architecture mismatch.")
        print("Please check if the model was trained with the same architecture.")
        return {}
    
    model.eval()
    
    # Test datasets - include all major datasets
    datasets = [
        {'name': 'DS1 Test', 'X_path': 'data/X_test_860.npy', 'y_path': 'data/y_test.npy'},
        {'name': 'DS1 Training', 'X_path': 'data/X_train_860.npy', 'y_path': 'data/y_train.npy'},
        {'name': 'DS3 Overlay', 'X_path': 'data/X_overlayed_860.npy', 'y_path': 'data/y_overlayed.npy'},
        {'name': 'CCNY May 2022', 'X_path': 'data/X_our_slab_size860.npy', 'y_path': None},
        {'name': 'CCNY June 2022', 'X_path': 'data/X_our_slab_size860.npy', 'y_path': None},  # Will be split
        {'name': 'CCNY Nov 2023', 'X_path': 'data/nov2023_non_resampled.npy', 'y_path': None},
    ]
    
    results = {}
    
    for dataset_info in datasets:
        dataset_name = dataset_info['name']
        X_path = dataset_info['X_path']
        y_path = dataset_info['y_path']
        
        print(f"\n=== Testing on {dataset_name} ===")
        
        try:
            # Handle different dataset types
            if y_path is not None:
                # Supervised datasets (DS1, DS3)
                test_dataset = MulticlassImpactEchoDataset([X_path], y_path=[y_path], array_size=860)
                test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False, num_workers=2)
                print(f"Testing supervised dataset on {len(test_dataset)} samples...")
                
                # Get predictions
                (accuracy, predictions, total_unc, epistemic_unc, 
                 aleatoric_unc, confidences, targets, alphas) = evaluate_multiclass_evidential_classifier(model, test_loader, device)
                
            else:
                # Unsupervised datasets (CCNY) - handle specially
                print(f"Testing unsupervised dataset: {dataset_name}")
                
                if 'May' in dataset_name or 'June' in dataset_name:
                    # CCNY May/June data (combined file)
                    X_may, X_june = load_ccny_sep2022_data_into_torch_tensor(device, X_path)
                    
                    if 'May' in dataset_name:
                        X_data = X_may
                        print(f"Testing CCNY May data: {len(X_data)} samples")
                    else:  # June
                        X_data = X_june
                        print(f"Testing CCNY June data: {len(X_data)} samples")
                        
                elif 'Nov' in dataset_name:
                    # CCNY Nov 2023 data
                    X_data = load_ccny_nov2023_data_into_torch_tensor2(device=device, X_path=X_path)
                    print(f"Testing CCNY Nov 2023 data: {len(X_data)} samples")
                else:
                    print(f"Skipping unknown unsupervised dataset: {dataset_name}")
                    continue
                
                # Get predictions for unsupervised data
                with torch.no_grad():
                    prob, epistemic, aleatoric, total_unc, confidence, alpha_sum = model.predict_with_uncertainty(X_data)
                
                # Create dummy targets and calculate dummy accuracy
                targets = torch.zeros(len(X_data), dtype=torch.long)  # Dummy targets
                accuracy = 0.0  # No accuracy for unsupervised
                
                # Format outputs to match supervised format (ensure proper shapes)
                predictions = prob  # Should be (1, n_samples, n_classes)
                epistemic_unc = epistemic  # Should be (1, n_samples, 1)
                aleatoric_unc = aleatoric  # Should be (1, n_samples, 1)
                total_unc = total_unc  # Should be (1, n_samples, 1)
                alphas = alpha_sum  # Should be (1, n_samples, 1)
                
                # Fix confidence shape to match expected format
                if confidence.dim() == 1:
                    # If confidence is 1D, expand to match expected shape for concatenation
                    confidences = confidence.unsqueeze(0)  # (n_samples,) -> (1, n_samples)
                else:
                    confidences = confidence
            
            if y_path is not None:
                print(f"Accuracy: {accuracy:.2f}%")
            else:
                print("Unsupervised dataset - no accuracy calculated")
            
            print(f"Mean Total Uncertainty: {total_unc.mean():.6f}")
            print(f"Mean Epistemic Uncertainty: {epistemic_unc.mean():.6f}")
            print(f"Mean Aleatoric Uncertainty: {aleatoric_unc.mean():.6f}")
            print(f"Mean Confidence: {confidences.mean():.6f}")
            print(f"Mean Evidence Strength: {alphas.mean():.6f}")
            
            # Class distribution for supervised datasets
            if y_path is not None:
                targets_np = targets.detach().cpu().numpy()
                pred_classes = torch.argmax(predictions.squeeze(0), dim=1).detach().cpu().numpy()
                print(f"True class distribution: {np.bincount(targets_np, minlength=5)}")
                print(f"Predicted class distribution: {np.bincount(pred_classes, minlength=5)}")
                print(f"Sample predictions shape: {pred_classes.shape}, targets shape: {targets_np.shape}")
            
            results[dataset_name] = {
                'accuracy': accuracy,
                'predictions': predictions,
                'total_uncertainties': total_unc,
                'epistemic_uncertainties': epistemic_unc,
                'aleatoric_uncertainties': aleatoric_unc,
                'confidences': confidences,
                'targets': targets,
                'alphas': alphas
            }
            
            # Generate defect maps and uncertainty maps for this dataset
            print(f"\n🗺️  Generating defect maps and uncertainty maps for {dataset_name}...")
            create_multiclass_defect_maps(results[dataset_name], dataset_name)
            
        except FileNotFoundError as e:
            print(f"Dataset not found for {dataset_name}: {e}")
            continue
        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    return results


def evaluate_multiclass_evidential_classifier(model, test_loader, device):
    """
    Evaluate multi-class evidential model with comprehensive uncertainty analysis
    """
    model.eval()
    correct = 0
    total = 0
    
    all_predictions = []
    all_total_unc = []
    all_epistemic_unc = []
    all_aleatoric_unc = []
    all_confidences = []
    all_targets = []
    all_alphas = []
    
    with torch.no_grad():
        for data in test_loader:
            X = data[0].to(device, dtype=torch.float)
            labels = data[1].to(device, dtype=torch.int).long()
            X = X.view(X.size(0), 1, X.size(1))
            
            prob, epistemic, aleatoric, total_unc, confidence, alpha_sum = model.predict_with_uncertainty(X)
            predicted = torch.argmax(prob.squeeze(0), dim=1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_predictions.append(prob.cpu())
            all_total_unc.append(total_unc.cpu())
            all_epistemic_unc.append(epistemic.cpu())
            all_aleatoric_unc.append(aleatoric.cpu())
            all_confidences.append(confidence.cpu())
            all_targets.append(labels.cpu())
            all_alphas.append(alpha_sum.cpu())
    
    accuracy = 100.0 * correct / total
    
    predictions = torch.cat(all_predictions, dim=1)
    total_uncertainties = torch.cat(all_total_unc, dim=1)
    epistemic_uncertainties = torch.cat(all_epistemic_unc, dim=1)
    aleatoric_uncertainties = torch.cat(all_aleatoric_unc, dim=1)
    confidences = torch.cat([conf.squeeze() for conf in all_confidences], dim=0)
    targets = torch.cat(all_targets, dim=0)
    alphas = torch.cat(all_alphas, dim=1)
    
    return (accuracy, predictions, total_uncertainties, epistemic_uncertainties, 
            aleatoric_uncertainties, confidences, targets, alphas)


def create_multiclass_defect_maps(results, dataset_name, experiment_name=None, model_name=None):
    """
    Create defect maps and uncertainty maps for multiclass evidential model
    """
    if experiment_name is None:
        experiment_name = default_experiment_name
    if model_name is None:
        model_name = default_model_name
    
    # Extract data
    predictions = results['predictions']
    total_unc = results['total_uncertainties']
    epistemic_unc = results['epistemic_uncertainties']
    aleatoric_unc = results['aleatoric_uncertainties']
    confidences = results['confidences']
    alphas = results['alphas']
    targets = results['targets']
    
    # Convert to numpy
    pred_probs = predictions.squeeze(0).cpu().numpy()  # Remove sequence dim
    total_unc_np = total_unc.squeeze(0).squeeze(-1).cpu().numpy()  # Remove extra dims
    epistemic_unc_np = epistemic_unc.squeeze(0).squeeze(-1).cpu().numpy()
    aleatoric_unc_np = aleatoric_unc.squeeze(0).squeeze(-1).cpu().numpy()
    confidences_np = confidences.squeeze().cpu().numpy() if confidences.dim() > 1 else confidences.cpu().numpy()
    alphas_np = alphas.squeeze(0).squeeze(-1).cpu().numpy()
    targets_np = targets.cpu().numpy()  # Keep for potential future use
    
    # Get predictions and dominant class probability
    pred_classes = np.argmax(pred_probs, axis=1)
    dominant_class_prob = np.max(pred_probs, axis=1)  # Probability of predicted class
    
    n_samples = len(pred_classes)
    print(f"Creating multiclass defect maps for {dataset_name} with {n_samples} samples...")
    
    # Determine spatial arrangement based on dataset
    dataset_shapes = {
        252: (9, 28),           # DS1: 252 samples = 9×28 spatial grid
        256: (9, 28),           # DS3 Overlay: same as DS1 = 9×28 spatial grid (with padding)
        1178: (31, 38),         # CCNY May 2022: 1178 samples = 31×38 grid
        646: (19, 34),          # CCNY June 2022: 646 samples = 19×34 grid  
        1496: (44, 34),         # CCNY Nov 2023: 1496 samples = 44×34 grid
    }
    
    # Get the correct spatial shape
    if n_samples in dataset_shapes:
        shape = dataset_shapes[n_samples]
        print(f"✅ Using exact spatial shape: {shape[0]}×{shape[1]} for {dataset_name}")
    else:
        # Fallback logic based on dataset name
        if 'DS1' in dataset_name.upper() or 'TEST' in dataset_name.upper():
            shape = (9, 28)
        elif 'MAY' in dataset_name.upper():
            shape = (31, 38)
        elif 'JUNE' in dataset_name.upper():
            shape = (19, 34)
        elif 'NOV' in dataset_name.upper():
            shape = (44, 34)
        elif 'OVERLAY' in dataset_name.upper() or 'DS3' in dataset_name.upper():
            shape = (9, 28)
        else:
            grid_size = int(np.sqrt(n_samples))
            shape = (grid_size, grid_size)
        print(f"🔍 Using fallback spatial shape: {shape[0]}×{shape[1]} for {dataset_name}")
    
    expected_samples = shape[0] * shape[1]
    
    # Handle size mismatch
    if n_samples != expected_samples:
        print(f"⚠️  Size mismatch: {n_samples} samples vs {expected_samples} expected ({shape})")
        if n_samples < expected_samples:
            pad_size = expected_samples - n_samples
            print(f"🔧 Padding {pad_size} missing spatial locations with NaN")
            
            pred_classes = np.pad(pred_classes, (0, pad_size), mode='constant', constant_values=-1)
            dominant_class_prob = np.pad(dominant_class_prob, (0, pad_size), mode='constant', constant_values=np.nan)
            epistemic_unc_np = np.pad(epistemic_unc_np, (0, pad_size), mode='constant', constant_values=np.nan)
            aleatoric_unc_np = np.pad(aleatoric_unc_np, (0, pad_size), mode='constant', constant_values=np.nan)
            total_unc_np = np.pad(total_unc_np, (0, pad_size), mode='constant', constant_values=np.nan)
            confidences_np = np.pad(confidences_np, (0, pad_size), mode='constant', constant_values=np.nan)
            alphas_np = np.pad(alphas_np, (0, pad_size), mode='constant', constant_values=np.nan)
        else:
            print(f"✂️  Truncating to first {expected_samples} samples to fit spatial grid")
            pred_classes = pred_classes[:expected_samples]
            dominant_class_prob = dominant_class_prob[:expected_samples]
            epistemic_unc_np = epistemic_unc_np[:expected_samples]
            aleatoric_unc_np = aleatoric_unc_np[:expected_samples]
            total_unc_np = total_unc_np[:expected_samples]
            confidences_np = confidences_np[:expected_samples]
            alphas_np = alphas_np[:expected_samples]
    
    # Reshape to spatial grids
    try:
        prediction_map = pred_classes.reshape(shape)
        dominant_prob_map = dominant_class_prob.reshape(shape)
        epistemic_map = epistemic_unc_np.reshape(shape)
        aleatoric_map = aleatoric_unc_np.reshape(shape)
        total_uncertainty_map = total_unc_np.reshape(shape)
        confidence_map = confidences_np.reshape(shape)
        evidence_map = alphas_np.reshape(shape)
        
        print(f"✅ Successfully created spatial maps with shape {shape}")
        
        # Generate individual defect maps
        save_multiclass_individual_defect_maps(
            prediction_map, dominant_prob_map, epistemic_map, aleatoric_map,
            total_uncertainty_map, confidence_map, evidence_map,
            dataset_name, shape, experiment_name, model_name
        )
        
        # Generate comprehensive spatial uncertainty maps (2×3 grid)
        create_multiclass_spatial_uncertainty_maps(
            prediction_map, dominant_prob_map, epistemic_map, aleatoric_map,
            total_uncertainty_map, confidence_map, evidence_map,
            dataset_name, shape, experiment_name, model_name
        )
        
        print(f"✅ All multiclass defect maps generated for {dataset_name}")
        
    except ValueError as e:
        print(f"❌ Error creating multiclass defect maps for {dataset_name}: {e}")
        print(f"Data shapes: predictions={pred_classes.shape}, expected reshape to {shape}")


def save_multiclass_individual_defect_maps(prediction_map, dominant_prob_map, epistemic_map, aleatoric_map, 
                                          total_uncertainty_map, confidence_map, evidence_map, 
                                          dataset_name, shape, experiment_name=None, model_name=None):
    """
    Save individual baseline-style defect maps for multiclass evidential model
    """
    if experiment_name is None:
        experiment_name = default_experiment_name
    if model_name is None:
        model_name = default_model_name
    
    safe_name = dataset_name.lower().replace(" ", "_").replace("-", "_")
    
    print(f"🎨 Generating individual multiclass defect maps for {dataset_name} (Shape: {shape[0]}×{shape[1]})...")
    
    # Calculate aspect ratio for rectangular maps with square pixels
    aspect_ratio = shape[1] / shape[0]  # width / height
    fig_width = 12
    fig_height = fig_width / aspect_ratio  # Adjust height to maintain rectangular shape
    
    # 1. Multiclass Classification map - DEFECT MAP with proper aspect ratio
    plt.figure(figsize=(fig_width, fig_height))
    im = plt.imshow(prediction_map, cmap='tab10', interpolation='nearest', aspect=1.0, vmin=0, vmax=4)
    plt.colorbar(im, label='Predicted Class', shrink=0.8, ticks=[0,1,2,3,4])
    plt.title(f'Multiclass Evidential Classification - {dataset_name}\\nSpatial Shape: {shape[0]}×{shape[1]} (5 Classes)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Spatial X Position')
    plt.ylabel('Spatial Y Position')
    plt.savefig(f'new_uncertainty_results/{experiment_name}/{model_name}_multiclass_{safe_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Dominant Class Probability map - CONFIDENCE MAP with proper aspect ratio
    plt.figure(figsize=(fig_width, fig_height))
    plt.imshow(dominant_prob_map, cmap='viridis', interpolation='hamming', aspect=1.0)
    plt.colorbar(label='Dominant Class Probability', shrink=0.8)
    plt.title(f'Multiclass Dominant Class Probability - {dataset_name}\\nProbability of Predicted Class', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Spatial X Position')
    plt.ylabel('Spatial Y Position')
    plt.savefig(f'new_uncertainty_results/{experiment_name}/{model_name}_multiclass_dominant_prob_{safe_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Epistemic uncertainty map - MODEL UNCERTAINTY DEFECT MAP with proper aspect ratio
    plt.figure(figsize=(fig_width, fig_height))
    plt.imshow(epistemic_map, cmap='Reds_r', interpolation='hamming', aspect=1.0)
    plt.colorbar(label='Epistemic Uncertainty (Higher=Darker)', shrink=0.8)
    plt.title(f'Multiclass Epistemic Uncertainty - {dataset_name}\\nModel Uncertainty (Darker = Higher Uncertainty)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Spatial X Position')
    plt.ylabel('Spatial Y Position')
    plt.savefig(f'new_uncertainty_results/{experiment_name}/{model_name}_multiclass_epistemic_{safe_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Aleatoric uncertainty map - DATA UNCERTAINTY DEFECT MAP with proper aspect ratio
    plt.figure(figsize=(fig_width, fig_height))
    plt.imshow(aleatoric_map, cmap='Blues_r', interpolation='hamming', aspect=1.0)
    plt.colorbar(label='Aleatoric Uncertainty (Higher=Darker)', shrink=0.8)
    plt.title(f'Multiclass Aleatoric Uncertainty - {dataset_name}\\nData Uncertainty (Darker = Higher Uncertainty)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Spatial X Position')
    plt.ylabel('Spatial Y Position')
    plt.savefig(f'new_uncertainty_results/{experiment_name}/{model_name}_multiclass_aleatoric_{safe_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Total uncertainty map - COMBINED UNCERTAINTY DEFECT MAP with proper aspect ratio
    plt.figure(figsize=(fig_width, fig_height))
    plt.imshow(total_uncertainty_map, cmap='Purples_r', interpolation='hamming', aspect=1.0)
    plt.colorbar(label='Total Uncertainty (Higher=Darker)', shrink=0.8)
    plt.title(f'Multiclass Total Uncertainty - {dataset_name}\\nCombined Uncertainty (Darker = Higher Uncertainty)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Spatial X Position')
    plt.ylabel('Spatial Y Position')
    plt.savefig(f'new_uncertainty_results/{experiment_name}/{model_name}_multiclass_total_uncertainty_{safe_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Confidence map - CONFIDENCE DEFECT MAP with proper aspect ratio
    plt.figure(figsize=(fig_width, fig_height))
    plt.imshow(confidence_map, cmap='Spectral', interpolation='hamming', aspect=1.0)
    plt.colorbar(label='Confidence (Higher=More Confident)', shrink=0.8)
    plt.title(f'Multiclass Confidence Map - {dataset_name}\\nPrediction Confidence', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Spatial X Position')
    plt.ylabel('Spatial Y Position')
    plt.savefig(f'new_uncertainty_results/{experiment_name}/{model_name}_multiclass_confidence_{safe_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 7. Evidence strength map - EVIDENCE DEFECT MAP with proper aspect ratio
    plt.figure(figsize=(fig_width, fig_height))
    plt.imshow(evidence_map, cmap='viridis', interpolation='hamming', aspect=1.0)
    plt.colorbar(label='Evidence Strength (Higher=Stronger Evidence)', shrink=0.8)
    plt.title(f'Multiclass Evidence Strength - {dataset_name}\\nEvidence Strength (Alpha Sum)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Spatial X Position')
    plt.ylabel('Spatial Y Position')
    plt.savefig(f'new_uncertainty_results/{experiment_name}/{model_name}_multiclass_evidence_{safe_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Individual multiclass defect maps saved for {dataset_name}:")
    print(f"  ✓ {model_name}_multiclass_{safe_name}.png (5-Class Classification)")
    print(f"  ✓ {model_name}_multiclass_dominant_prob_{safe_name}.png (Dominant Class Probability)")
    print(f"  ✓ {model_name}_multiclass_epistemic_{safe_name}.png (Model Uncertainty)")
    print(f"  ✓ {model_name}_multiclass_aleatoric_{safe_name}.png (Data Uncertainty)")
    print(f"  ✓ {model_name}_multiclass_total_uncertainty_{safe_name}.png (Total Uncertainty)")
    print(f"  ✓ {model_name}_multiclass_confidence_{safe_name}.png (Confidence)")
    print(f"  ✓ {model_name}_multiclass_evidence_{safe_name}.png (Evidence Strength)")


def create_multiclass_spatial_uncertainty_maps(prediction_map, dominant_prob_map, epistemic_map, aleatoric_map,
                                              total_uncertainty_map, confidence_map, evidence_map,
                                              dataset_name, shape, experiment_name=None, model_name=None):
    """
    Create comprehensive spatial uncertainty maps (2×3 grid) for multiclass evidential model
    """
    if experiment_name is None:
        experiment_name = default_experiment_name
    if model_name is None:
        model_name = default_model_name
    
    print(f"🗺️  Creating comprehensive multiclass spatial maps for {dataset_name}...")
    
    # Create comprehensive SPATIAL uncertainty visualization (2×3 grid)
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'Multiclass Evidential Uncertainty Maps - {dataset_name} (Spatial: {shape[0]}×{shape[1]})', 
                fontsize=18, fontweight='bold')
    
    # 1. Multiclass Classification map
    im1 = axes[0, 0].imshow(prediction_map, cmap='tab10', interpolation='nearest', aspect='equal', vmin=0, vmax=4)
    axes[0, 0].set_title(f'5-Class Classification\\n(0-4: Different Defect Types)', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Spatial X Position')
    axes[0, 0].set_ylabel('Spatial Y Position')
    cbar1 = plt.colorbar(im1, ax=axes[0, 0], shrink=0.8)
    cbar1.set_label('Predicted Class (0-4)')
    
    # 2. Dominant Class Probability map
    im2 = axes[0, 1].imshow(dominant_prob_map, cmap='viridis', interpolation='hamming', aspect='equal')
    axes[0, 1].set_title(f'Dominant Class Probability\\n(Confidence in Prediction)', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Spatial X Position')
    axes[0, 1].set_ylabel('Spatial Y Position')
    cbar2 = plt.colorbar(im2, ax=axes[0, 1], shrink=0.8)
    cbar2.set_label('Probability')
    
    # 3. Epistemic uncertainty map - model uncertainty
    im3 = axes[0, 2].imshow(epistemic_map, cmap='Reds_r', interpolation='hamming', aspect='equal')
    axes[0, 2].set_title(f'Epistemic Uncertainty\\n(Model Uncertainty)', fontsize=12, fontweight='bold')
    axes[0, 2].set_xlabel('Spatial X Position')
    axes[0, 2].set_ylabel('Spatial Y Position')
    cbar3 = plt.colorbar(im3, ax=axes[0, 2], shrink=0.8)
    cbar3.set_label('Epistemic Uncertainty (Higher=Darker)')
    
    # 4. Aleatoric uncertainty map - data uncertainty
    im4 = axes[1, 0].imshow(aleatoric_map, cmap='Blues_r', interpolation='hamming', aspect='equal')
    axes[1, 0].set_title(f'Aleatoric Uncertainty\\n(Data Uncertainty)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Spatial X Position')
    axes[1, 0].set_ylabel('Spatial Y Position')
    cbar4 = plt.colorbar(im4, ax=axes[1, 0], shrink=0.8)
    cbar4.set_label('Aleatoric Uncertainty (Higher=Darker)')
    
    # 5. Total uncertainty map - combined uncertainty
    im5 = axes[1, 1].imshow(total_uncertainty_map, cmap='Purples_r', interpolation='hamming', aspect='equal')
    axes[1, 1].set_title(f'Total Uncertainty\\n(Combined)', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Spatial X Position')
    axes[1, 1].set_ylabel('Spatial Y Position')
    cbar5 = plt.colorbar(im5, ax=axes[1, 1], shrink=0.8)
    cbar5.set_label('Total Uncertainty (Higher=Darker)')
    
    # 6. Evidence strength map
    im6 = axes[1, 2].imshow(evidence_map, cmap='viridis', interpolation='hamming', aspect='equal')
    axes[1, 2].set_title(f'Evidence Strength\\n(Alpha Sum)', fontsize=12, fontweight='bold')
    axes[1, 2].set_xlabel('Spatial X Position')
    axes[1, 2].set_ylabel('Spatial Y Position')
    cbar6 = plt.colorbar(im6, ax=axes[1, 2], shrink=0.8)
    cbar6.set_label('Evidence Strength')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Save comprehensive multiclass spatial maps
    safe_dataset_name = dataset_name.lower().replace(" ", "_").replace("-", "_")
    comprehensive_filename = f'{model_name}_multiclass_spatial_maps_{safe_dataset_name}.png'
    plt.savefig(f'new_uncertainty_results/{experiment_name}/{comprehensive_filename}', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Comprehensive multiclass spatial uncertainty maps saved: {comprehensive_filename}")
    print(f"  📁 File: new_uncertainty_results/{experiment_name}/{comprehensive_filename}")
    print(f"  🗺️  Shape: {shape[0]}×{shape[1]} representing real spatial locations")
    print(f"  📊 6 maps: 5-Class Prediction | Dominant Prob | Epistemic | Aleatoric | Total | Evidence")


if __name__ == '__main__':
    # Configuration
    experiment_name = "evidential_multiclass_v1"
    model_name = "evidential_multiclass_v1"
    model_path = f'weights/{model_name}.pth'
    
    print("=== Multi-class Evidential Uncertainty Analysis ===")
    print(f"🚀 Running experiment: {experiment_name} with model: {model_name}")
    print("🚀 Analyzing 5-class defect classification with uncertainty quantification!\n")
    
    # Create experiment directory
    import os
    os.makedirs(f'new_uncertainty_results/{experiment_name}', exist_ok=True)
    
    try:
        # Test model on datasets and get results
        print("1. Running multi-class inference on test datasets...")
        dataset_results = test_multiclass_evidential_model_on_datasets(model_path)
        
        if dataset_results:
            print("✓ Multi-class inference complete! Now generating analysis...\n")
            
            # Analyze results for each dataset
            for dataset_name, results in dataset_results.items():
                print(f"\n=== Analyzing {dataset_name} Results ===")
                
                # Save results in the expected format
                torch.save({
                    'predictions': results['predictions'],
                    'total_uncertainties': results['total_uncertainties'],
                    'epistemic_uncertainties': results['epistemic_uncertainties'],
                    'aleatoric_uncertainties': results['aleatoric_uncertainties'],
                    'confidences': results['confidences'],
                    'targets': results['targets'],
                    'alphas': results['alphas'],
                    'accuracy': results['accuracy'],
                    'model_config': {
                        'num_classes': 5,
                        'model_type': 'multiclass_evidential'
                    }
                }, f'weights/{model_name}_{dataset_name.lower().replace(" ", "_")}_results.pth')
                
                # Analyze the results
                analyze_multiclass_evidential_results(
                    f'weights/{model_name}_{dataset_name.lower().replace(" ", "_")}_results.pth', 
                    dataset_name
                )
            
            print(f"\n✓ Multi-class evidential analysis complete!")
            
        else:
            print("❌ No results generated - check model file")
            
    except FileNotFoundError:
        print(f"❌ Model file not found: {model_path}")
        print("Please train the multi-class model first using train_evidential_multiclass.py")
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n=== Multi-class Analysis Complete ===")
    print("✓ All multi-class uncertainty visualizations completed successfully!")
    print(f"\n📁 Generated Analysis Files:")
    print(f"✓ new_uncertainty_results/{experiment_name}/{model_name}_multiclass_uncertainty_distributions.png")
    print(f"✓ new_uncertainty_results/{experiment_name}/{model_name}_multiclass_confusion_matrix.png")
    print(f"✓ new_uncertainty_results/{experiment_name}/{model_name}_classwise_uncertainty_comparison.png")
    print(f"✓ new_uncertainty_results/{experiment_name}/{model_name}_multiclass_confidence_analysis.png")
    print(f"✓ weights/{model_name}_*_results.pth (detailed results for each dataset)")
    
    print("\n🎉 Multi-class evidential uncertainty analysis complete!")
    print("🔬 This analysis provides insights into:")
    print("  • 5-class defect type classification performance")
    print("  • Uncertainty quantification for each defect type")
    print("  • Model confidence calibration across classes")
    print("  • Class-specific uncertainty patterns")
    
    # Ensure all plots are closed
    plt.close('all')
    print("\n✓ All matplotlib resources cleaned up.")
    print("🔴 Multi-class analysis finished. Safe to exit.")


def create_multiclass_defect_maps(results, dataset_name, experiment_name=None, model_name=None):
    """
    Create defect maps and uncertainty maps for multiclass evidential model
    """
    if experiment_name is None:
        experiment_name = default_experiment_name
    if model_name is None:
        model_name = default_model_name
    
    # Extract data
    predictions = results['predictions']
    total_unc = results['total_uncertainties']
    epistemic_unc = results['epistemic_uncertainties']
    aleatoric_unc = results['aleatoric_uncertainties']
    confidences = results['confidences']
    alphas = results['alphas']
    targets = results['targets']
    
    # Convert to numpy
    pred_probs = predictions.squeeze(0).cpu().numpy()  # Remove sequence dim
    total_unc_np = total_unc.squeeze(0).squeeze(-1).cpu().numpy()  # Remove extra dims
    epistemic_unc_np = epistemic_unc.squeeze(0).squeeze(-1).cpu().numpy()
    aleatoric_unc_np = aleatoric_unc.squeeze(0).squeeze(-1).cpu().numpy()
    confidences_np = confidences.squeeze().cpu().numpy() if confidences.dim() > 1 else confidences.cpu().numpy()
    alphas_np = alphas.squeeze(0).squeeze(-1).cpu().numpy()
    targets_np = targets.cpu().numpy()  # Keep for potential future use
    
    # Get predictions and dominant class probability
    pred_classes = np.argmax(pred_probs, axis=1)
    dominant_class_prob = np.max(pred_probs, axis=1)  # Probability of predicted class
    
    n_samples = len(pred_classes)
    print(f"Creating multiclass defect maps for {dataset_name} with {n_samples} samples...")
    
    # Determine spatial arrangement based on dataset
    dataset_shapes = {
        252: (9, 28),           # DS1: 252 samples = 9×28 spatial grid
        256: (9, 28),           # DS3 Overlay: same as DS1 = 9×28 spatial grid (with padding)
        1178: (31, 38),         # CCNY May 2022: 1178 samples = 31×38 grid
        646: (19, 34),          # CCNY June 2022: 646 samples = 19×34 grid  
        1496: (44, 34),         # CCNY Nov 2023: 1496 samples = 44×34 grid
    }
    
    # Get the correct spatial shape
    if n_samples in dataset_shapes:
        shape = dataset_shapes[n_samples]
        print(f"✅ Using exact spatial shape: {shape[0]}×{shape[1]} for {dataset_name}")
    else:
        # Fallback logic based on dataset name
        if 'DS1' in dataset_name.upper() or 'TEST' in dataset_name.upper():
            shape = (9, 28)
        elif 'MAY' in dataset_name.upper():
            shape = (31, 38)
        elif 'JUNE' in dataset_name.upper():
            shape = (19, 34)
        elif 'NOV' in dataset_name.upper():
            shape = (44, 34)
        elif 'OVERLAY' in dataset_name.upper() or 'DS3' in dataset_name.upper():
            shape = (9, 28)
        else:
            grid_size = int(np.sqrt(n_samples))
            shape = (grid_size, grid_size)
        print(f"🔍 Using fallback spatial shape: {shape[0]}×{shape[1]} for {dataset_name}")
    
    expected_samples = shape[0] * shape[1]
    
    # Handle size mismatch
    if n_samples != expected_samples:
        print(f"⚠️  Size mismatch: {n_samples} samples vs {expected_samples} expected ({shape})")
        if n_samples < expected_samples:
            pad_size = expected_samples - n_samples
            print(f"🔧 Padding {pad_size} missing spatial locations with NaN")
            
            pred_classes = np.pad(pred_classes, (0, pad_size), mode='constant', constant_values=-1)
            dominant_class_prob = np.pad(dominant_class_prob, (0, pad_size), mode='constant', constant_values=np.nan)
            epistemic_unc_np = np.pad(epistemic_unc_np, (0, pad_size), mode='constant', constant_values=np.nan)
            aleatoric_unc_np = np.pad(aleatoric_unc_np, (0, pad_size), mode='constant', constant_values=np.nan)
            total_unc_np = np.pad(total_unc_np, (0, pad_size), mode='constant', constant_values=np.nan)
            confidences_np = np.pad(confidences_np, (0, pad_size), mode='constant', constant_values=np.nan)
            alphas_np = np.pad(alphas_np, (0, pad_size), mode='constant', constant_values=np.nan)
        else:
            print(f"✂️  Truncating to first {expected_samples} samples to fit spatial grid")
            pred_classes = pred_classes[:expected_samples]
            dominant_class_prob = dominant_class_prob[:expected_samples]
            epistemic_unc_np = epistemic_unc_np[:expected_samples]
            aleatoric_unc_np = aleatoric_unc_np[:expected_samples]
            total_unc_np = total_unc_np[:expected_samples]
            confidences_np = confidences_np[:expected_samples]
            alphas_np = alphas_np[:expected_samples]
    
    # Reshape to spatial grids
    try:
        prediction_map = pred_classes.reshape(shape)
        dominant_prob_map = dominant_class_prob.reshape(shape)
        epistemic_map = epistemic_unc_np.reshape(shape)
        aleatoric_map = aleatoric_unc_np.reshape(shape)
        total_uncertainty_map = total_unc_np.reshape(shape)
        confidence_map = confidences_np.reshape(shape)
        evidence_map = alphas_np.reshape(shape)
        
        print(f"✅ Successfully created spatial maps with shape {shape}")
        
        # Generate individual defect maps
        save_multiclass_individual_defect_maps(
            prediction_map, dominant_prob_map, epistemic_map, aleatoric_map,
            total_uncertainty_map, confidence_map, evidence_map,
            dataset_name, shape, experiment_name, model_name
        )
        
        # Generate comprehensive spatial uncertainty maps (2×3 grid)
        create_multiclass_spatial_uncertainty_maps(
            prediction_map, dominant_prob_map, epistemic_map, aleatoric_map,
            total_uncertainty_map, confidence_map, evidence_map,
            dataset_name, shape, experiment_name, model_name
        )
        
        print(f"✅ All multiclass defect maps generated for {dataset_name}")
        
    except ValueError as e:
        print(f"❌ Error creating multiclass defect maps for {dataset_name}: {e}")
        print(f"Data shapes: predictions={pred_classes.shape}, expected reshape to {shape}")


def save_multiclass_individual_defect_maps(prediction_map, dominant_prob_map, epistemic_map, aleatoric_map, 
                                          total_uncertainty_map, confidence_map, evidence_map, 
                                          dataset_name, shape, experiment_name=None, model_name=None):
    """
    Save individual baseline-style defect maps for multiclass evidential model
    """
    if experiment_name is None:
        experiment_name = default_experiment_name
    if model_name is None:
        model_name = default_model_name
    
    safe_name = dataset_name.lower().replace(" ", "_").replace("-", "_")
    
    print(f"🎨 Generating individual multiclass defect maps for {dataset_name} (Shape: {shape[0]}×{shape[1]})...")
    
    # Calculate aspect ratio for rectangular maps with square pixels
    aspect_ratio = shape[1] / shape[0]  # width / height
    fig_width = 12
    fig_height = fig_width / aspect_ratio  # Adjust height to maintain rectangular shape
    
    # Define multiclass colormap for predictions (5 classes)
    # class_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']  # 5 distinct colors
    # class_names = ['Solid Concrete', 'Defect Type 1', 'Defect Type 2', 'Defect Type 3', 'Defect Type 4']
    
    # 1. Multiclass Classification map - DEFECT MAP with proper aspect ratio
    plt.figure(figsize=(fig_width, fig_height))
    im = plt.imshow(prediction_map, cmap='tab10', interpolation='nearest', aspect=1.0, vmin=0, vmax=4)
    plt.colorbar(im, label='Predicted Class', shrink=0.8, ticks=[0,1,2,3,4])
    plt.title(f'Multiclass Evidential Classification - {dataset_name}\nSpatial Shape: {shape[0]}×{shape[1]} (5 Classes)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Spatial X Position')
    plt.ylabel('Spatial Y Position')
    plt.savefig(f'new_uncertainty_results/{experiment_name}/{model_name}_multiclass_{safe_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Dominant Class Probability map - CONFIDENCE MAP with proper aspect ratio
    plt.figure(figsize=(fig_width, fig_height))
    plt.imshow(dominant_prob_map, cmap='viridis', interpolation='hamming', aspect=1.0)
    plt.colorbar(label='Dominant Class Probability', shrink=0.8)
    plt.title(f'Multiclass Dominant Class Probability - {dataset_name}\nProbability of Predicted Class', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Spatial X Position')
    plt.ylabel('Spatial Y Position')
    plt.savefig(f'new_uncertainty_results/{experiment_name}/{model_name}_multiclass_dominant_prob_{safe_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Epistemic uncertainty map - MODEL UNCERTAINTY DEFECT MAP with proper aspect ratio
    plt.figure(figsize=(fig_width, fig_height))
    plt.imshow(epistemic_map, cmap='Reds_r', interpolation='hamming', aspect=1.0)
    plt.colorbar(label='Epistemic Uncertainty (Higher=Darker)', shrink=0.8)
    plt.title(f'Multiclass Epistemic Uncertainty - {dataset_name}\nModel Uncertainty (Darker = Higher Uncertainty)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Spatial X Position')
    plt.ylabel('Spatial Y Position')
    plt.savefig(f'new_uncertainty_results/{experiment_name}/{model_name}_multiclass_epistemic_{safe_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Aleatoric uncertainty map - DATA UNCERTAINTY DEFECT MAP with proper aspect ratio
    plt.figure(figsize=(fig_width, fig_height))
    plt.imshow(aleatoric_map, cmap='Blues_r', interpolation='hamming', aspect=1.0)
    plt.colorbar(label='Aleatoric Uncertainty (Higher=Darker)', shrink=0.8)
    plt.title(f'Multiclass Aleatoric Uncertainty - {dataset_name}\nData Uncertainty (Darker = Higher Uncertainty)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Spatial X Position')
    plt.ylabel('Spatial Y Position')
    plt.savefig(f'new_uncertainty_results/{experiment_name}/{model_name}_multiclass_aleatoric_{safe_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Total uncertainty map - COMBINED UNCERTAINTY DEFECT MAP with proper aspect ratio
    plt.figure(figsize=(fig_width, fig_height))
    plt.imshow(total_uncertainty_map, cmap='Purples_r', interpolation='hamming', aspect=1.0)
    plt.colorbar(label='Total Uncertainty (Higher=Darker)', shrink=0.8)
    plt.title(f'Multiclass Total Uncertainty - {dataset_name}\nCombined Uncertainty (Darker = Higher Uncertainty)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Spatial X Position')
    plt.ylabel('Spatial Y Position')
    plt.savefig(f'new_uncertainty_results/{experiment_name}/{model_name}_multiclass_total_uncertainty_{safe_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 6. Confidence map - CONFIDENCE DEFECT MAP with proper aspect ratio
    plt.figure(figsize=(fig_width, fig_height))
    plt.imshow(confidence_map, cmap='Spectral', interpolation='hamming', aspect=1.0)
    plt.colorbar(label='Confidence (Higher=More Confident)', shrink=0.8)
    plt.title(f'Multiclass Confidence Map - {dataset_name}\nPrediction Confidence', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Spatial X Position')
    plt.ylabel('Spatial Y Position')
    plt.savefig(f'new_uncertainty_results/{experiment_name}/{model_name}_multiclass_confidence_{safe_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 7. Evidence strength map - EVIDENCE DEFECT MAP with proper aspect ratio
    plt.figure(figsize=(fig_width, fig_height))
    plt.imshow(evidence_map, cmap='viridis', interpolation='hamming', aspect=1.0)
    plt.colorbar(label='Evidence Strength (Higher=Stronger Evidence)', shrink=0.8)
    plt.title(f'Multiclass Evidence Strength - {dataset_name}\nEvidence Strength (Alpha Sum)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Spatial X Position')
    plt.ylabel('Spatial Y Position')
    plt.savefig(f'new_uncertainty_results/{experiment_name}/{model_name}_multiclass_evidence_{safe_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Individual multiclass defect maps saved for {dataset_name}:")
    print(f"  ✓ {model_name}_multiclass_{safe_name}.png (5-Class Classification)")
    print(f"  ✓ {model_name}_multiclass_dominant_prob_{safe_name}.png (Dominant Class Probability)")
    print(f"  ✓ {model_name}_multiclass_epistemic_{safe_name}.png (Model Uncertainty)")
    print(f"  ✓ {model_name}_multiclass_aleatoric_{safe_name}.png (Data Uncertainty)")
    print(f"  ✓ {model_name}_multiclass_total_uncertainty_{safe_name}.png (Total Uncertainty)")
    print(f"  ✓ {model_name}_multiclass_confidence_{safe_name}.png (Confidence)")
    print(f"  ✓ {model_name}_multiclass_evidence_{safe_name}.png (Evidence Strength)")


def create_multiclass_spatial_uncertainty_maps(prediction_map, dominant_prob_map, epistemic_map, aleatoric_map,
                                              total_uncertainty_map, confidence_map, evidence_map,
                                              dataset_name, shape, experiment_name=None, model_name=None):
    """
    Create comprehensive spatial uncertainty maps (2×3 grid) for multiclass evidential model
    """
    if experiment_name is None:
        experiment_name = default_experiment_name
    if model_name is None:
        model_name = default_model_name
    
    print(f"🗺️  Creating comprehensive multiclass spatial maps for {dataset_name}...")
    
    # Create comprehensive SPATIAL uncertainty visualization (2×3 grid)
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'Multiclass Evidential Uncertainty Maps - {dataset_name} (Spatial: {shape[0]}×{shape[1]})', 
                fontsize=18, fontweight='bold')
    
    # 1. Multiclass Classification map
    im1 = axes[0, 0].imshow(prediction_map, cmap='tab10', interpolation='nearest', aspect='equal', vmin=0, vmax=4)
    axes[0, 0].set_title(f'5-Class Classification\n(0-4: Different Defect Types)', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Spatial X Position')
    axes[0, 0].set_ylabel('Spatial Y Position')
    cbar1 = plt.colorbar(im1, ax=axes[0, 0], shrink=0.8)
    cbar1.set_label('Predicted Class (0-4)')
    
    # 2. Dominant Class Probability map
    im2 = axes[0, 1].imshow(dominant_prob_map, cmap='viridis', interpolation='hamming', aspect='equal')
    axes[0, 1].set_title(f'Dominant Class Probability\n(Confidence in Prediction)', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Spatial X Position')
    axes[0, 1].set_ylabel('Spatial Y Position')
    cbar2 = plt.colorbar(im2, ax=axes[0, 1], shrink=0.8)
    cbar2.set_label('Probability')
    
    # 3. Epistemic uncertainty map - model uncertainty
    im3 = axes[0, 2].imshow(epistemic_map, cmap='Reds_r', interpolation='hamming', aspect='equal')
    axes[0, 2].set_title(f'Epistemic Uncertainty\n(Model Uncertainty)', fontsize=12, fontweight='bold')
    axes[0, 2].set_xlabel('Spatial X Position')
    axes[0, 2].set_ylabel('Spatial Y Position')
    cbar3 = plt.colorbar(im3, ax=axes[0, 2], shrink=0.8)
    cbar3.set_label('Epistemic Uncertainty (Higher=Darker)')
    
    # 4. Aleatoric uncertainty map - data uncertainty
    im4 = axes[1, 0].imshow(aleatoric_map, cmap='Blues_r', interpolation='hamming', aspect='equal')
    axes[1, 0].set_title(f'Aleatoric Uncertainty\n(Data Uncertainty)', fontsize=12, fontweight='bold')
    axes[1, 0].set_xlabel('Spatial X Position')
    axes[1, 0].set_ylabel('Spatial Y Position')
    cbar4 = plt.colorbar(im4, ax=axes[1, 0], shrink=0.8)
    cbar4.set_label('Aleatoric Uncertainty (Higher=Darker)')
    
    # 5. Total uncertainty map - combined uncertainty
    im5 = axes[1, 1].imshow(total_uncertainty_map, cmap='Purples_r', interpolation='hamming', aspect='equal')
    axes[1, 1].set_title(f'Total Uncertainty\n(Combined)', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Spatial X Position')
    axes[1, 1].set_ylabel('Spatial Y Position')
    cbar5 = plt.colorbar(im5, ax=axes[1, 1], shrink=0.8)
    cbar5.set_label('Total Uncertainty (Higher=Darker)')
    
    # 6. Evidence strength map
    im6 = axes[1, 2].imshow(evidence_map, cmap='viridis', interpolation='hamming', aspect='equal')
    axes[1, 2].set_title(f'Evidence Strength\n(Alpha Sum)', fontsize=12, fontweight='bold')
    axes[1, 2].set_xlabel('Spatial X Position')
    axes[1, 2].set_ylabel('Spatial Y Position')
    cbar6 = plt.colorbar(im6, ax=axes[1, 2], shrink=0.8)
    cbar6.set_label('Evidence Strength')
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    
    # Save comprehensive multiclass spatial maps
    safe_dataset_name = dataset_name.lower().replace(" ", "_").replace("-", "_")
    comprehensive_filename = f'{model_name}_multiclass_spatial_maps_{safe_dataset_name}.png'
    plt.savefig(f'new_uncertainty_results/{experiment_name}/{comprehensive_filename}', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Comprehensive multiclass spatial uncertainty maps saved: {comprehensive_filename}")
    print(f"  📁 File: new_uncertainty_results/{experiment_name}/{comprehensive_filename}")
    print(f"  🗺️  Shape: {shape[0]}×{shape[1]} representing real spatial locations")
    print(f"  📊 6 maps: 5-Class Prediction | Dominant Prob | Epistemic | Aleatoric | Total | Evidence")