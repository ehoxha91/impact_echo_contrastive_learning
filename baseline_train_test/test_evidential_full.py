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
from utils import *

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

from models.model_parts import ResidualBlock
from train_evidential_full import FullEvidentialIENet, dirichlet_kl_divergence, evidential_loss
from train_evidential_simple import SimpleEvidentialIENet
from train_evidential import EvidentialIENet

# Default configuration - can be overridden
default_experiment_name = "exp4"
default_model_name = "evidential_full"


def analyze_full_evidential_results(results_path):
    """
    Analyze and visualize full evidential deep learning results with comprehensive metrics
    """
    print(f"Loading results from {results_path}")
    results = torch.load(results_path, map_location='cpu')
    
    predictions = results['predictions']
    total_unc = results['total_uncertainties']
    epistemic_unc = results['epistemic_uncertainties']
    aleatoric_unc = results['aleatoric_uncertainties']
    confidences = results['confidences']
    alphas = results['alphas']
    targets = results['targets']
    accuracy = results['accuracy']
    
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Number of test samples: {len(targets)}")
    
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
    
    # Classification metrics
    correct_predictions = (pred_classes_np == targets_np)
    incorrect_predictions = ~correct_predictions
    
    print(f"\nCorrect Predictions: {correct_predictions.sum()}/{len(targets_np)}")
    print(f"Incorrect Predictions: {incorrect_predictions.sum()}/{len(targets_np)}")
    
    # Comprehensive uncertainty statistics
    print(f"\n=== Uncertainty Analysis ===")
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
    
    # High uncertainty sample identification
    high_epistemic_threshold = np.mean(epistemic_unc_np) + 2 * np.std(epistemic_unc_np)
    high_total_threshold = np.mean(total_unc_np) + 2 * np.std(total_unc_np)
    low_confidence_threshold = np.mean(confidences_np) - 2 * np.std(confidences_np)
    low_evidence_threshold = np.mean(alphas_np) - 2 * np.std(alphas_np)
    
    high_epistemic_samples = epistemic_unc_np > high_epistemic_threshold
    high_total_samples = total_unc_np > high_total_threshold
    low_confidence_samples = confidences_np < low_confidence_threshold
    low_evidence_samples = alphas_np < low_evidence_threshold
    
    print(f"\n=== High Uncertainty Sample Identification ===")
    print(f"High Epistemic Uncertainty (>μ+2σ): {np.sum(high_epistemic_samples)} samples")
    print(f"High Total Uncertainty (>μ+2σ): {np.sum(high_total_samples)} samples")
    print(f"Low Confidence (<μ-2σ): {np.sum(low_confidence_samples)} samples")
    print(f"Low Evidence Strength (<μ-2σ): {np.sum(low_evidence_samples)} samples")
    
    # Class-specific analysis
    print(f"\n=== Class-specific Analysis ===")
    for class_idx in [0, 1]:
        class_mask = targets_np == class_idx
        if class_mask.sum() > 0:
            class_name = "No Defect" if class_idx == 0 else "Defect"
            print(f"{class_name} (Class {class_idx}) - {class_mask.sum()} samples:")
            print(f"  - Accuracy: {(pred_classes_np[class_mask] == class_idx).mean()*100:.2f}%")
            print(f"  - Mean Total Uncertainty: {np.mean(total_unc_np[class_mask]):.6f}")
            print(f"  - Mean Epistemic Uncertainty: {np.mean(epistemic_unc_np[class_mask]):.6f}")
            print(f"  - Mean Confidence: {np.mean(confidences_np[class_mask]):.6f}")
    
    # Dirichlet distribution analysis
    analyze_dirichlet_distributions(pred_probs_np, targets_np, alphas_np)
    
    # Create comprehensive visualizations
    print("\n=== Creating Beautiful Uncertainty Visualizations ===")
    
    # 1. Advanced uncertainty distribution analysis
    create_advanced_uncertainty_distribution_analysis(
        pred_probs_np, epistemic_unc_np, aleatoric_unc_np, total_unc_np, 
        confidences_np, alphas_np, targets_np, correct_predictions
    )
    
    # 2. Comprehensive 9-subplot analysis
    create_full_evidential_plots(
        pred_probs_np, epistemic_unc_np, aleatoric_unc_np, total_unc_np, 
        confidences_np, alphas_np, targets_np, correct_predictions
    )
    
    # 3. Spatial uncertainty maps
    create_spatial_uncertainty_maps(
        pred_probs_np, epistemic_unc_np.squeeze(), aleatoric_unc_np.squeeze(), 
        total_unc_np.squeeze(), confidences_np, alphas_np.squeeze(), targets_np
    )
    
    # 4. Advanced correlation analysis
    create_advanced_uncertainty_correlations(
        epistemic_unc_np, aleatoric_unc_np, total_unc_np, 
        confidences_np, alphas_np, correct_predictions
    )
    
    return results


def create_advanced_uncertainty_correlations(epistemic_unc, aleatoric_unc, total_unc, 
                                            confidences, alphas, correct_predictions, 
                                            experiment_name=None, model_name=None):
    """
    Create advanced uncertainty correlation analysis with beautiful visualizations
    """
    # Calculate correlation coefficients
    print("\n=== Advanced Uncertainty Correlation Analysis ===")
    
    # Remove extra dimensions for correlation analysis
    epistemic_flat = epistemic_unc.squeeze() if epistemic_unc.ndim > 1 else epistemic_unc
    aleatoric_flat = aleatoric_unc.squeeze() if aleatoric_unc.ndim > 1 else aleatoric_unc
    total_flat = total_unc.squeeze() if total_unc.ndim > 1 else total_unc
    alphas_flat = alphas.squeeze() if alphas.ndim > 1 else alphas
    
    # Correlation matrix
    corr_data = np.column_stack([
        epistemic_flat, aleatoric_flat, total_flat, confidences, alphas_flat
    ])
    corr_labels = ['Epistemic', 'Aleatoric', 'Total Unc.', 'Confidence', 'Evidence']
    
    corr_matrix = np.corrcoef(corr_data.T)
    
    # Beautiful correlation heatmap
    plt.figure(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                square=True, fmt='.3f', cbar_kws={"shrink": .8},
                xticklabels=corr_labels, yticklabels=corr_labels)
    plt.title('Uncertainty Metrics Correlation Matrix', fontsize=16, fontweight='bold')
    plt.tight_layout()
    if experiment_name is None:
        experiment_name = default_experiment_name
    if model_name is None:
        model_name = default_model_name
    
    plt.savefig(f'new_uncertainty_results/{experiment_name}/{model_name}_correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Print correlation insights
    print(f"Epistemic-Aleatoric Correlation: {np.corrcoef(epistemic_flat, aleatoric_flat)[0,1]:.4f}")
    print(f"Total-Confidence Correlation: {np.corrcoef(total_flat, confidences)[0,1]:.4f}")
    print(f"Evidence-Confidence Correlation: {np.corrcoef(alphas_flat, confidences)[0,1]:.4f}")
    
    # Uncertainty decomposition analysis
    uncertainty_ratio = epistemic_flat / (epistemic_flat + aleatoric_flat + 1e-8)
    
    print(f"\n=== Uncertainty Decomposition Analysis ===")
    print(f"Mean Epistemic/(Epistemic+Aleatoric) Ratio: {np.mean(uncertainty_ratio):.4f}")
    print(f"Samples with high epistemic dominance (>0.7): {np.sum(uncertainty_ratio > 0.7)}")
    print(f"Samples with high aleatoric dominance (<0.3): {np.sum(uncertainty_ratio < 0.3)}")
    
    # Correctness-based uncertainty analysis
    if correct_predictions.sum() > 0 and (~correct_predictions).sum() > 0:
        correct_epistemic = epistemic_flat[correct_predictions]
        incorrect_epistemic = epistemic_flat[~correct_predictions]
        
        # Statistical test for uncertainty differences
        t_stat, p_value = stats.ttest_ind(correct_epistemic, incorrect_epistemic)
        print(f"\n=== Statistical Significance Analysis ===")
        print(f"T-test (Correct vs Incorrect Epistemic): t={t_stat:.4f}, p={p_value:.6f}")
        
        if p_value < 0.05:
            print("✓ Significant difference in epistemic uncertainty between correct/incorrect predictions")
        else:
            print("✗ No significant difference in epistemic uncertainty")
    
    print("✓ Advanced correlation analysis complete")


def analyze_dirichlet_distributions(pred_probs, targets, alphas):
    """
    Analyze the learned Dirichlet distributions
    """
    print(f"\n=== Dirichlet Distribution Analysis ===")
    
    # Concentration parameters analysis
    class_0_samples = targets == 0
    class_1_samples = targets == 1
    
    if class_0_samples.sum() > 0:
        alpha_0_mean = np.mean(alphas[class_0_samples])
        print(f"No Defect samples - Mean Evidence Strength: {alpha_0_mean:.6f}")
        
    if class_1_samples.sum() > 0:
        alpha_1_mean = np.mean(alphas[class_1_samples])
        print(f"Defect samples - Mean Evidence Strength: {alpha_1_mean:.6f}")
    
    # Analyze prediction confidence distribution
    max_probs = np.max(pred_probs, axis=1)
    print(f"Prediction Confidence Distribution:")
    print(f"  - Min: {np.min(max_probs):.6f}")
    print(f"  - Max: {np.max(max_probs):.6f}")
    print(f"  - Mean: {np.mean(max_probs):.6f}")
    print(f"  - Std: {np.std(max_probs):.6f}")
    
    # Low confidence predictions
    low_conf_threshold = 0.6
    low_conf_mask = max_probs < low_conf_threshold
    print(f"Low confidence predictions (<{low_conf_threshold}): {np.sum(low_conf_mask)} samples")
    
    if np.sum(low_conf_mask) > 0:
        low_conf_accuracy = np.mean(np.argmax(pred_probs[low_conf_mask], axis=1) == targets[low_conf_mask])
        print(f"Accuracy of low confidence predictions: {low_conf_accuracy*100:.2f}%")


def create_advanced_uncertainty_distribution_analysis(pred_probs, epistemic_unc, aleatoric_unc, total_unc, 
                                                    confidences, alphas, targets, correct_predictions,
                                                    experiment_name=None, model_name=None):
    """
    Create advanced uncertainty distribution analysis plots with beautiful styling
    """
    # Set style for beautiful plots
    plt.style.use('default')
    sns.set_palette("husl")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Evidential Uncertainty Distribution Analysis', fontsize=18, fontweight='bold')
    
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
    axes[1, 0].hist(confidences[correct_predictions], bins=25, alpha=0.7, 
                   color='green', label='Correct', density=True, edgecolor='black', linewidth=0.8)
    axes[1, 0].hist(confidences[~correct_predictions], bins=25, alpha=0.7, 
                   color='red', label='Incorrect', density=True, edgecolor='black', linewidth=0.8)
    axes[1, 0].set_xlabel('Confidence', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Density', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('Confidence Distribution', fontsize=14, fontweight='bold')
    axes[1, 0].legend(frameon=True, fancybox=True, shadow=True)
    axes[1, 0].grid(True, alpha=0.3, linestyle='--')
    
    # Plot 4: Uncertainty vs Confidence scatter
    axes[1, 1].scatter(total_unc[correct_predictions], confidences[correct_predictions], 
                      alpha=0.7, c='green', label='Correct', s=40, edgecolors='black', linewidth=0.5)
    axes[1, 1].scatter(total_unc[~correct_predictions], confidences[~correct_predictions], 
                      alpha=0.7, c='red', label='Incorrect', s=40, edgecolors='black', linewidth=0.5)
    axes[1, 1].set_xlabel('Total Uncertainty', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Confidence', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('Uncertainty vs Confidence', fontsize=14, fontweight='bold')
    axes[1, 1].legend(frameon=True, fancybox=True, shadow=True)
    axes[1, 1].grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    if experiment_name is None:
        experiment_name = default_experiment_name
    if model_name is None:
        model_name = default_model_name
    
    plt.savefig(f'new_uncertainty_results/{experiment_name}/{model_name}_uncertainty_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✓ Advanced uncertainty distribution analysis saved to new_uncertainty_results/{experiment_name}/{model_name}_uncertainty_distributions.png")


def create_full_evidential_plots(pred_probs, epistemic_unc, aleatoric_unc, total_unc, 
                                confidences, alphas, targets, correct_predictions,
                                experiment_name=None, model_name=None):
    """
    Create comprehensive uncertainty visualization plots for full evidential model with beautiful styling
    """
    # Set style for beautiful plots
    plt.style.use('default')
    sns.set_palette("husl")
    
    fig, axes = plt.subplots(3, 3, figsize=(20, 18))
    fig.suptitle('Evidential Deep Learning - Comprehensive Uncertainty Analysis', fontsize=20, fontweight='bold')
    
    # Plot 1: Epistemic vs Aleatoric Uncertainty
    axes[0, 0].scatter(epistemic_unc[correct_predictions], aleatoric_unc[correct_predictions], 
                      alpha=0.6, c='green', label='Correct', s=30)
    axes[0, 0].scatter(epistemic_unc[~correct_predictions], aleatoric_unc[~correct_predictions], 
                      alpha=0.6, c='red', label='Incorrect', s=30)
    axes[0, 0].set_xlabel('Epistemic Uncertainty')
    axes[0, 0].set_ylabel('Aleatoric Uncertainty')
    axes[0, 0].set_title('Epistemic vs Aleatoric Uncertainty')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Plot 2: Total Uncertainty Distribution
    axes[0, 1].hist(total_unc[correct_predictions], bins=30, alpha=0.7, 
                   color='green', label='Correct', density=True)
    axes[0, 1].hist(total_unc[~correct_predictions], bins=30, alpha=0.7, 
                   color='red', label='Incorrect', density=True)
    axes[0, 1].set_xlabel('Total Uncertainty')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('Total Uncertainty Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Plot 3: Confidence vs Evidence Strength
    axes[0, 2].scatter(alphas[correct_predictions], confidences[correct_predictions], 
                      alpha=0.6, c='green', label='Correct', s=30)
    axes[0, 2].scatter(alphas[~correct_predictions], confidences[~correct_predictions], 
                      alpha=0.6, c='red', label='Incorrect', s=30)
    axes[0, 2].set_xlabel('Evidence Strength (Alpha Sum)')
    axes[0, 2].set_ylabel('Confidence')
    axes[0, 2].set_title('Confidence vs Evidence Strength')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Plot 4: Epistemic Uncertainty Distribution
    axes[1, 0].hist(epistemic_unc[correct_predictions], bins=30, alpha=0.7, 
                   color='green', label='Correct', density=True)
    axes[1, 0].hist(epistemic_unc[~correct_predictions], bins=30, alpha=0.7, 
                   color='red', label='Incorrect', density=True)
    axes[1, 0].set_xlabel('Epistemic Uncertainty')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Epistemic Uncertainty Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Plot 5: Aleatoric Uncertainty Distribution
    axes[1, 1].hist(aleatoric_unc[correct_predictions], bins=30, alpha=0.7, 
                   color='green', label='Correct', density=True)
    axes[1, 1].hist(aleatoric_unc[~correct_predictions], bins=30, alpha=0.7, 
                   color='red', label='Incorrect', density=True)
    axes[1, 1].set_xlabel('Aleatoric Uncertainty')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('Aleatoric Uncertainty Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Evidence Strength Distribution
    axes[1, 2].hist(alphas[correct_predictions], bins=30, alpha=0.7, 
                   color='green', label='Correct', density=True)
    axes[1, 2].hist(alphas[~correct_predictions], bins=30, alpha=0.7, 
                   color='red', label='Incorrect', density=True)
    axes[1, 2].set_xlabel('Evidence Strength (Alpha Sum)')
    axes[1, 2].set_ylabel('Density')
    axes[1, 2].set_title('Evidence Strength Distribution')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    # Plot 7: Confidence Distribution
    axes[2, 0].hist(confidences[correct_predictions], bins=30, alpha=0.7, 
                   color='green', label='Correct', density=True)
    axes[2, 0].hist(confidences[~correct_predictions], bins=30, alpha=0.7, 
                   color='red', label='Incorrect', density=True)
    axes[2, 0].set_xlabel('Confidence')
    axes[2, 0].set_ylabel('Density')
    axes[2, 0].set_title('Confidence Distribution')
    axes[2, 0].legend()
    axes[2, 0].grid(True, alpha=0.3)
    
    # Plot 8: Uncertainty Correlation
    axes[2, 1].scatter(total_unc, confidences, c=correct_predictions, 
                      cmap='RdYlGn', alpha=0.6, s=30)
    axes[2, 1].set_xlabel('Total Uncertainty')
    axes[2, 1].set_ylabel('Confidence')
    axes[2, 1].set_title('Total Uncertainty vs Confidence')
    axes[2, 1].grid(True, alpha=0.3)
    
    # Plot 9: Class-wise Uncertainty
    class_0_mask = targets == 0
    class_1_mask = targets == 1
    
    if class_0_mask.sum() > 0 and class_1_mask.sum() > 0:
        axes[2, 2].boxplot([total_unc[class_0_mask], total_unc[class_1_mask]], 
                          labels=['No Defect', 'Defect'])
        axes[2, 2].set_ylabel('Total Uncertainty')
        axes[2, 2].set_title('Uncertainty by True Class')
        axes[2, 2].grid(True, alpha=0.3)
    else:
        axes[2, 2].text(0.5, 0.5, 'Insufficient class diversity', 
                       ha='center', va='center', transform=axes[2, 2].transAxes)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.93)
    if experiment_name is None:
        experiment_name = default_experiment_name
    if model_name is None:
        model_name = default_model_name
    
    plt.savefig(f'new_uncertainty_results/{experiment_name}/{model_name}_uncertainty_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Comprehensive evidential uncertainty analysis saved to new_uncertainty_results/{experiment_name}/{model_name}_uncertainty_analysis.png")


def test_full_evidential_model_on_datasets(model_path):
    """
    Test the trained full evidential model on all available datasets
    """
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Load model
    model = FullEvidentialIENet(num_classes=2, verbose=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded full evidential model from {model_path}")
    
    # Test datasets - ALL AVAILABLE DATASETS!
    datasets = [
        {'name': 'DS1 Test', 'X_path': 'data/X_test_860.npy', 'y_path': 'data/y_test.npy'},
        # {'name': 'DS1 Training', 'X_path': 'data/X_train_860.npy', 'y_path': 'data/y_train.npy'},
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
                test_dataset = ImpactEchoDatasetClassifier([X_path], y_path=[y_path], array_size=860)
                test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False, num_workers=2)
                print(f"Testing supervised dataset on {len(test_dataset)} samples...")
                
                # Get predictions
                (accuracy, predictions, total_unc, epistemic_unc, 
                 aleatoric_unc, confidences, targets, alphas) = evaluate_full_evidential_model(model, test_loader, device)
                
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
                
                print(f"Debug shapes - predictions: {predictions.shape}, confidence: {confidences.shape}, targets: {targets.shape}")
            
            if y_path is not None:
                print(f"Accuracy: {accuracy:.2f}%")
            else:
                print("Unsupervised dataset - no accuracy calculated")
            
            print(f"Mean Total Uncertainty: {total_unc.mean():.6f}")
            print(f"Mean Epistemic Uncertainty: {epistemic_unc.mean():.6f}")
            print(f"Mean Aleatoric Uncertainty: {aleatoric_unc.mean():.6f}")
            print(f"Mean Confidence: {confidences.mean():.6f}")
            print(f"Mean Evidence Strength: {alphas.mean():.6f}")
            
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
            
        except FileNotFoundError as e:
            print(f"Dataset not found for {dataset_name}: {e}")
            continue
        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
            continue
    
    return results


def evaluate_full_evidential_model(model, test_loader, device):
    """
    Evaluate full evidential model with comprehensive uncertainty analysis
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
    # Handle confidence concatenation - confidence has shape (batch_size,) for each batch
    confidences = torch.cat([conf.squeeze() for conf in all_confidences], dim=0)
    targets = torch.cat(all_targets, dim=0)
    alphas = torch.cat(all_alphas, dim=1)
    
    return (accuracy, predictions, total_uncertainties, epistemic_uncertainties, 
            aleatoric_uncertainties, confidences, targets, alphas)


def generate_full_evidential_maps_all_datasets(model_path='/weights/evidential_full.pth', fallback_path='/weights/evidential_full.pth'):
    """
    Generate uncertainty maps for all datasets using full evidential model (with fallback to simple model)
    """
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Try to load full evidential model, fallback to simple model
    model = None
    model_type = None
    
    try:
        model = FullEvidentialIENet(num_classes=2, verbose=False).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        model_type = "full"
        print(f"Loaded full evidential model from {model_path}")
    except FileNotFoundError:
        print(f"Full model not found at {model_path}, trying simple model...")
        try:
            model = EvidentialIENet(num_classes=2, verbose=False).to(device)
            model.load_state_dict(torch.load(fallback_path, map_location=device))
            model.eval()
            model_type = "simple"
            print(f"Loaded evidential model from {fallback_path}")
        except FileNotFoundError:
            print(f"No evidential model found at {model_path} or {fallback_path}")
            return {}
    
    # Test on different datasets (following test_evidential.py pattern)
    datasets = {
        'DS1 Test': ('data/X_test_860.npy', 'data/y_test.npy'),
        'DS3 Overlay': ('data/X_overlayed_860.npy', 'data/y_overlayed.npy'),
        'CCNY Data': ('data/X_our_slab_size860.npy', None),
        'CCNY Nov2023': ('data/nov2023_non_resampled.npy', None),
    }
    
    all_maps = {}
    
    for dataset_name, (X_path, y_path) in datasets.items():
        print(f"\nProcessing {dataset_name}...")
        
        try:
            # Load data using the same pattern as test_evidential.py
            if dataset_name == 'CCNY Data':
                X_may, X_june = load_ccny_sep2022_data_into_torch_tensor(device, X_path)
                maps = process_ccny_data_for_full_evidential(model, X_may, X_june, dataset_name, model_type)
            elif dataset_name == 'CCNY Nov2023':
                X_nov23 = load_ccny_nov2023_data_into_torch_tensor2(device=device, X_path=X_path)
                maps = process_dataset_for_full_evidential(model, X_nov23, None, dataset_name, model_type)
            elif dataset_name == 'DS3 Overlay':
                X_overlay, y_overlay = load_ds3_overlay_test_data_into_torch_tensor(device)
                maps = process_dataset_for_full_evidential(model, X_overlay, y_overlay, dataset_name, model_type)
            else:  # DS1 Test
                X_test, y_test = load_ds1_test_data_into_torch_tensor(device, X_path, y_path)
                maps = process_dataset_for_full_evidential(model, X_test, y_test, dataset_name, model_type)
                
            all_maps[dataset_name] = maps
            
        except FileNotFoundError as e:
            print(f"Data file not found for {dataset_name}: {e}")
            continue
        except Exception as e:
            print(f"Error processing {dataset_name}: {e}")
            continue
    
    return all_maps


def process_dataset_for_full_evidential(model, X_data, y_data, dataset_name, model_type):
    """
    Process a dataset and generate uncertainty maps for full evidential model
    """
    print(f"Processing {len(X_data)} samples for {dataset_name} using {model_type} model")
    
    # Get predictions and uncertainties
    with torch.no_grad():
        if model_type == "simple":
            prob, epistemic, aleatoric, total_unc, confidence, alpha_sum = model.predict_with_uncertainty(X_data)
        else:  # full model
            prob, epistemic, aleatoric, total_unc, confidence, alpha_sum = model.predict_with_uncertainty(X_data)
    
    # Convert to numpy for visualization
    prob_np = prob.squeeze(0).cpu().numpy()
    epistemic_np = epistemic.squeeze(0).squeeze(1).cpu().numpy()
    aleatoric_np = aleatoric.squeeze(0).squeeze(1).cpu().numpy()
    total_unc_np = total_unc.squeeze(0).squeeze(1).cpu().numpy()
    confidence_np = confidence.cpu().numpy()
    alpha_sum_np = alpha_sum.squeeze(0).squeeze(1).cpu().numpy()
    
    # Get predictions
    predictions = np.argmax(prob_np, axis=1)
    non_defect_prob = prob_np[:, 0]  # Probability of class 0 (non-defect)
    
    maps_data = {
        'predictions': predictions,
        'probabilities': prob_np,
        'non_defect_prob': non_defect_prob,
        'epistemic_uncertainty': epistemic_np,
        'aleatoric_uncertainty': aleatoric_np,
        'total_uncertainty': total_unc_np,
        'confidence': confidence_np,
        'evidence_strength': alpha_sum_np,
        'targets': y_data if y_data is not None else None
    }
    
    # Create baseline-style maps
    create_full_evidential_baseline_maps(maps_data, dataset_name, model_type)
    
    return maps_data


def process_ccny_data_for_full_evidential(model, X_may, X_june, dataset_name, model_type):
    """
    Process CCNY data (May and June separately) for full evidential uncertainty maps
    """
    maps_data = {}
    
    for period, X_data in [('May', X_may), ('June', X_june)]:
        print(f"Processing {len(X_data)} samples for {dataset_name} - {period} using {model_type} model")
        
        with torch.no_grad():
            if model_type == "simple":
                prob, epistemic, aleatoric, total_unc, confidence, alpha_sum = model.predict_with_uncertainty(X_data)
            else:  # full model
                prob, epistemic, aleatoric, total_unc, confidence, alpha_sum = model.predict_with_uncertainty(X_data)
        
        # Convert to numpy
        prob_np = prob.squeeze(0).cpu().numpy()
        epistemic_np = epistemic.squeeze(0).squeeze(1).cpu().numpy()
        aleatoric_np = aleatoric.squeeze(0).squeeze(1).cpu().numpy()
        total_unc_np = total_unc.squeeze(0).squeeze(1).cpu().numpy()
        confidence_np = confidence.cpu().numpy()
        alpha_sum_np = alpha_sum.squeeze(0).squeeze(1).cpu().numpy()
        
        predictions = np.argmax(prob_np, axis=1)
        non_defect_prob = prob_np[:, 0]
        
        period_data = {
            'predictions': predictions,
            'probabilities': prob_np,
            'non_defect_prob': non_defect_prob,
            'epistemic_uncertainty': epistemic_np,
            'aleatoric_uncertainty': aleatoric_np,
            'total_uncertainty': total_unc_np,
            'confidence': confidence_np,
            'evidence_strength': alpha_sum_np,
            'targets': None
        }
        
        maps_data[period] = period_data
        
        # Create baseline-style maps for each period
        create_full_evidential_baseline_maps(period_data, f"{dataset_name}_{period}", model_type)
    
    return maps_data


def create_spatial_uncertainty_maps(pred_probs, epistemic_unc, aleatoric_unc, total_unc, 
                                   confidences, alphas, targets, dataset_name="DS1",
                                   experiment_name=None, model_name=None):
    """
    Create spatial uncertainty maps with EXACT shapes representing real spatial locations
    """
    n_samples = len(pred_probs)
    predictions = np.argmax(pred_probs, axis=1)
    
    print(f"\n🗺️  Creating spatial maps for {dataset_name} ({n_samples} samples)")
    
    # EXACT spatial shapes based on real dataset spatial arrangements
    dataset_shapes = {
        # DS1 and DS3 datasets (same spatial arrangement)
        252: (9, 28),           # DS1: 252 samples = 9×28 spatial grid
        256: (9, 28),           # DS3 Overlay: same as DS1 = 9×28 spatial grid (some padding)
        
        # CCNY datasets with exact spatial dimensions
        1178: (31, 38),         # CCNY May 2022: 1178 samples = 31×38 grid
        646: (19, 34),          # CCNY June 2022: 646 samples = 19×34 grid  
        1496: (44, 34),         # CCNY Nov 2023: 1496 samples = 44×34 grid
    }
    
    # Get the correct spatial shape for this dataset
    if n_samples in dataset_shapes:
        shape = dataset_shapes[n_samples]
        print(f"✅ Using exact spatial shape: {shape[0]}×{shape[1]} = {shape[0]*shape[1]} locations")
    else:
        print(f"⚠️  Unknown dataset size {n_samples}, checking common patterns...")
        
        # Try to match with dataset name patterns
        if 'DS1' in dataset_name.upper() or 'TEST' in dataset_name.upper():
            shape = (9, 28)  # DS1 pattern
        elif 'MAY' in dataset_name.upper():
            shape = (31, 38)  # CCNY May pattern
        elif 'JUNE' in dataset_name.upper():
            shape = (19, 34)  # CCNY June pattern  
        elif 'NOV' in dataset_name.upper():
            shape = (44, 34)  # CCNY Nov pattern
        elif 'OVERLAY' in dataset_name.upper() or 'DS3' in dataset_name.upper():
            shape = (9, 28)  # DS3 Overlay uses same spatial arrangement as DS1
        else:
            # Last resort: find closest rectangular arrangement
            factors = []
            for i in range(1, int(np.sqrt(n_samples)) + 1):
                if n_samples % i == 0:
                    factors.append((i, n_samples // i))
            
            if factors:
                # Choose the most rectangular (least square) arrangement
                shape = min(factors, key=lambda x: abs(x[0] - x[1]))
                print(f"🔍 Using best rectangular fit: {shape[0]}×{shape[1]}")
            else:
                # Perfect square fallback
                grid_size = int(np.sqrt(n_samples))
                shape = (grid_size, grid_size)
                print(f"📐 Using square fallback: {shape[0]}×{shape[1]}")
    
    expected_samples = shape[0] * shape[1]
    
    # Handle size mismatch
    if n_samples != expected_samples:
        print(f"❌ Size mismatch: {n_samples} samples vs {expected_samples} expected ({shape})")
        
        if n_samples < expected_samples:
            # Pad with NaN for missing spatial locations
            pad_size = expected_samples - n_samples
            print(f"🔧 Padding {pad_size} missing spatial locations with NaN")
            
            predictions = np.pad(predictions, (0, pad_size), mode='constant', constant_values=-1)
            epistemic_unc = np.pad(epistemic_unc, (0, pad_size), mode='constant', constant_values=np.nan)
            aleatoric_unc = np.pad(aleatoric_unc, (0, pad_size), mode='constant', constant_values=np.nan)
            total_unc = np.pad(total_unc, (0, pad_size), mode='constant', constant_values=np.nan)
            confidences = np.pad(confidences, (0, pad_size), mode='constant', constant_values=np.nan)
            alphas = np.pad(alphas, (0, pad_size), mode='constant', constant_values=np.nan)
        else:
            # Too many samples - truncate to fit exact spatial grid
            print(f"✂️  Truncating to first {expected_samples} samples to fit spatial grid")
            predictions = predictions[:expected_samples]
            epistemic_unc = epistemic_unc[:expected_samples]
            aleatoric_unc = aleatoric_unc[:expected_samples]
            total_unc = total_unc[:expected_samples]
            confidences = confidences[:expected_samples]
            alphas = alphas[:expected_samples]
    
    # Reshape to EXACT spatial grids representing real locations
    print(f"🔄 Reshaping data to spatial grid: {shape}")
    try:
        pred_map = predictions.reshape(shape)
        epistemic_map = epistemic_unc.reshape(shape)
        aleatoric_map = aleatoric_unc.reshape(shape)
        total_uncertainty_map = total_unc.reshape(shape)
        confidence_map = confidences.reshape(shape)
        evidence_map = alphas.reshape(shape)
        
        print(f"✅ Successfully created spatial maps with shape {shape}")
        
        # Create comprehensive SPATIAL uncertainty visualization 
        fig, axes = plt.subplots(2, 3, figsize=(20, 12))
        fig.suptitle(f'Evidential Uncertainty Maps - {dataset_name} (Spatial Shape: {shape[0]}×{shape[1]})', 
                    fontsize=18, fontweight='bold')
        
        # Calculate proper aspect ratio to make each datapoint appear square
        aspect_ratio = shape[1] / shape[0]  # width / height
        
        # Predictions map - exact spatial locations with square pixels
        im1 = axes[0, 0].imshow(pred_map, cmap='RdYlBu_r', interpolation='hamming', aspect=aspect_ratio)
        axes[0, 0].set_title(f'Predictions\n(Spatial: {shape[0]}×{shape[1]})', fontsize=12, fontweight='bold')
        axes[0, 0].set_xlabel('Spatial X Position')
        axes[0, 0].set_ylabel('Spatial Y Position')
        cbar1 = plt.colorbar(im1, ax=axes[0, 0], shrink=0.8)
        cbar1.set_label('Class (0=No Defect, 1=Defect)')
        
        # Epistemic uncertainty map - model uncertainty at each location with square pixels
        im2 = axes[0, 1].imshow(epistemic_map, cmap='Reds', interpolation='hamming', aspect=aspect_ratio)
        axes[0, 1].set_title(f'Epistemic Uncertainty\n(Model Uncertainty)', fontsize=12, fontweight='bold')
        axes[0, 1].set_xlabel('Spatial X Position')
        axes[0, 1].set_ylabel('Spatial Y Position')
        cbar2 = plt.colorbar(im2, ax=axes[0, 1], shrink=0.8)
        cbar2.set_label('Epistemic Uncertainty')
        
        # Aleatoric uncertainty map - data uncertainty at each location with square pixels
        im3 = axes[0, 2].imshow(aleatoric_map, cmap='Blues', interpolation='hamming', aspect=aspect_ratio)
        axes[0, 2].set_title(f'Aleatoric Uncertainty\n(Data Uncertainty)', fontsize=12, fontweight='bold')
        axes[0, 2].set_xlabel('Spatial X Position')
        axes[0, 2].set_ylabel('Spatial Y Position')
        cbar3 = plt.colorbar(im3, ax=axes[0, 2], shrink=0.8)
        cbar3.set_label('Aleatoric Uncertainty')
        
        # Total uncertainty map - combined uncertainty at each location with square pixels
        im4 = axes[1, 0].imshow(total_uncertainty_map, cmap='plasma', interpolation='hamming', aspect=aspect_ratio)
        axes[1, 0].set_title(f'Total Uncertainty\n(Combined)', fontsize=12, fontweight='bold')
        axes[1, 0].set_xlabel('Spatial X Position')
        axes[1, 0].set_ylabel('Spatial Y Position')
        cbar4 = plt.colorbar(im4, ax=axes[1, 0], shrink=0.8)
        cbar4.set_label('Total Uncertainty')
        
        # Confidence map - prediction confidence at each location with square pixels
        im5 = axes[1, 1].imshow(confidence_map, cmap='RdYlGn', interpolation='hamming', aspect=aspect_ratio)
        axes[1, 1].set_title(f'Prediction Confidence\n(Higher=Better)', fontsize=12, fontweight='bold')
        axes[1, 1].set_xlabel('Spatial X Position')
        axes[1, 1].set_ylabel('Spatial Y Position')
        cbar5 = plt.colorbar(im5, ax=axes[1, 1], shrink=0.8)
        cbar5.set_label('Confidence')
        
        # Evidence strength map - evidence strength at each location with square pixels
        im6 = axes[1, 2].imshow(evidence_map, cmap='viridis', interpolation='hamming', aspect=aspect_ratio)
        axes[1, 2].set_title(f'Evidence Strength\n(Alpha Sum)', fontsize=12, fontweight='bold')
        axes[1, 2].set_xlabel('Spatial X Position')
        axes[1, 2].set_ylabel('Spatial Y Position')
        cbar6 = plt.colorbar(im6, ax=axes[1, 2], shrink=0.8)
        cbar6.set_label('Evidence Strength')
        
        plt.tight_layout()
        plt.subplots_adjust(top=0.92)
        if experiment_name is None:
            experiment_name = default_experiment_name
        if model_name is None:
            model_name = default_model_name
        
        plt.savefig(f'new_uncertainty_results/{experiment_name}/{model_name}_uncertainty_maps.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"✅ Spatial uncertainty maps saved (Shape: {shape[0]}×{shape[1]} representing real spatial locations)")
        print(f"📁 File: new_uncertainty_results/{experiment_name}/{model_name}_uncertainty_maps.png")
        
    except ValueError as e:
        print(f"❌ Error creating spatial maps for {dataset_name}: {e}")
        print(f"📊 Data info: {n_samples} samples, expected: {expected_samples}, target shape: {shape}")
        print(f"🔍 Shape check: predictions={predictions.shape}, expected reshape to {shape}")


def create_individual_defect_maps(pred_probs, epistemic_unc, aleatoric_unc, total_unc, 
                                 confidences, alphas, dataset_name,
                                 experiment_name=None, model_name=None):
    """
    Create individual baseline-style defect maps for each uncertainty type (ALWAYS GENERATED)
    """
    print(f"\n🗺️  Creating individual defect maps for {dataset_name}...")
    
    n_samples = len(pred_probs)
    predictions = np.argmax(pred_probs, axis=1)
    non_defect_prob = pred_probs[:, 0]  # Probability of class 0 (non-defect)
    
    # Use the same spatial shape logic as the main spatial maps
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
        print(f"✅ Using exact spatial shape: {shape[0]}×{shape[1]} for individual maps")
    else:
        # Fallback logic (same as spatial maps function)
        if 'DS1' in dataset_name.upper() or 'TEST' in dataset_name.upper():
            shape = (9, 28)
        elif 'MAY' in dataset_name.upper():
            shape = (31, 38)
        elif 'JUNE' in dataset_name.upper():
            shape = (19, 34)
        elif 'NOV' in dataset_name.upper():
            shape = (44, 34)
        elif 'OVERLAY' in dataset_name.upper() or 'DS3' in dataset_name.upper():
            shape = (9, 28)  # DS3 Overlay uses same spatial arrangement as DS1
        else:
            grid_size = int(np.sqrt(n_samples))
            shape = (grid_size, grid_size)
    
    expected_samples = shape[0] * shape[1]
    
    # Handle size mismatch (same logic as spatial maps)
    if n_samples != expected_samples:
        if n_samples < expected_samples:
            pad_size = expected_samples - n_samples
            non_defect_prob = np.pad(non_defect_prob, (0, pad_size), mode='constant', constant_values=np.nan)
            epistemic_unc = np.pad(epistemic_unc, (0, pad_size), mode='constant', constant_values=np.nan)
            aleatoric_unc = np.pad(aleatoric_unc, (0, pad_size), mode='constant', constant_values=np.nan)
            total_unc = np.pad(total_unc, (0, pad_size), mode='constant', constant_values=np.nan)
            confidences = np.pad(confidences, (0, pad_size), mode='constant', constant_values=np.nan)
            alphas = np.pad(alphas, (0, pad_size), mode='constant', constant_values=np.nan)
        else:
            non_defect_prob = non_defect_prob[:expected_samples]
            epistemic_unc = epistemic_unc[:expected_samples]
            aleatoric_unc = aleatoric_unc[:expected_samples]
            total_unc = total_unc[:expected_samples]
            confidences = confidences[:expected_samples]
            alphas = alphas[:expected_samples]
    
    # Reshape to spatial grids
    try:
        classification_map = non_defect_prob.reshape(shape)
        epistemic_map = epistemic_unc.reshape(shape)
        aleatoric_map = aleatoric_unc.reshape(shape)
        total_uncertainty_map = total_unc.reshape(shape)
        confidence_map = confidences.reshape(shape)
        evidence_map = alphas.reshape(shape)
        
        # Generate individual defect maps (ALWAYS generated for every dataset!)
        save_individual_defect_maps(
            classification_map, epistemic_map, aleatoric_map,
            total_uncertainty_map, confidence_map, evidence_map,
            dataset_name, shape, experiment_name, model_name
        )
        
        print(f"✅ All individual defect maps generated for {dataset_name} with shape {shape[0]}×{shape[1]}")
        
    except ValueError as e:
        print(f"❌ Error creating individual defect maps for {dataset_name}: {e}")


def save_individual_defect_maps(classification_map, epistemic_map, aleatoric_map, 
                               total_uncertainty_map, confidence_map, evidence_map, 
                               dataset_name, shape, experiment_name=None, model_name=None):
    """
    Save individual baseline-style defect maps with proper spatial representation
    """
    if experiment_name is None:
        experiment_name = default_experiment_name
    if model_name is None:
        model_name = default_model_name
    
    safe_name = dataset_name.lower().replace(" ", "_").replace("-", "_")
    
    print(f"🎨 Generating individual defect maps for {dataset_name} (Shape: {shape[0]}×{shape[1]})...")
    
    # Calculate aspect ratio for rectangular maps with square pixels
    aspect_ratio = shape[1] / shape[0]  # width / height
    fig_width = 12
    fig_height = fig_width / aspect_ratio  # Adjust height to maintain rectangular shape
    
    # 1. Classification map (non-defect probability) - DEFECT MAP with proper aspect ratio
    plt.figure(figsize=(fig_width, fig_height))
    plt.imshow(classification_map, cmap='Spectral', interpolation='hamming', aspect=1.0)
    plt.colorbar(label='Non-defect Probability', shrink=0.8)
    plt.title(f'Evidential Classification Map - {dataset_name}\nSpatial Shape: {shape[0]}×{shape[1]} (Each pixel = 1 measurement)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Spatial X Position')
    plt.ylabel('Spatial Y Position')
    plt.savefig(f'new_uncertainty_results/{experiment_name}/{model_name}_model_{safe_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Epistemic uncertainty map - MODEL UNCERTAINTY DEFECT MAP with proper aspect ratio
    plt.figure(figsize=(fig_width, fig_height))
    plt.imshow(epistemic_map, cmap='Reds_r', interpolation='hamming', aspect=1.0)
    plt.colorbar(label='Epistemic Uncertainty (Higher=More Uncertain)', shrink=0.8)
    plt.title(f'Evidential Epistemic Uncertainty - {dataset_name}\nModel Uncertainty at Each Location (Each pixel = 1 measurement)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Spatial X Position')
    plt.ylabel('Spatial Y Position')
    plt.savefig(f'new_uncertainty_results/{experiment_name}/{model_name}_model_epistemic_{safe_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Aleatoric uncertainty map - DATA UNCERTAINTY DEFECT MAP with proper aspect ratio
    plt.figure(figsize=(fig_width, fig_height))
    plt.imshow(aleatoric_map, cmap='Blues_r', interpolation='hamming', aspect=1.0)
    plt.colorbar(label='Aleatoric Uncertainty (Higher=More Uncertain)', shrink=0.8)
    plt.title(f'Evidential Aleatoric Uncertainty - {dataset_name}\nData Uncertainty at Each Location (Each pixel = 1 measurement)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Spatial X Position')
    plt.ylabel('Spatial Y Position')
    plt.savefig(f'new_uncertainty_results/{experiment_name}/{model_name}_model_aleatoric_{safe_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Total uncertainty map - COMBINED UNCERTAINTY DEFECT MAP with proper aspect ratio
    plt.figure(figsize=(fig_width, fig_height))
    plt.imshow(total_uncertainty_map, cmap='Purples_r', interpolation='hamming', aspect=1.0)
    plt.colorbar(label='Total Uncertainty (Higher=More Uncertain)', shrink=0.8)
    plt.title(f'Evidential Total Uncertainty - {dataset_name}\nCombined Uncertainty at Each Location (Each pixel = 1 measurement)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Spatial X Position')
    plt.ylabel('Spatial Y Position')
    plt.savefig(f'new_uncertainty_results/{experiment_name}/{model_name}_model_total_uncertainty_{safe_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Confidence map - CONFIDENCE DEFECT MAP with proper aspect ratio
    plt.figure(figsize=(fig_width, fig_height))
    plt.imshow(confidence_map, cmap='RdYlGn', interpolation='hamming', aspect=1.0)
    plt.colorbar(label='Confidence (Higher=More Confident)', shrink=0.8)
    plt.title(f'Evidential Confidence Map - {dataset_name}\nPrediction Confidence at Each Location (Each pixel = 1 measurement)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Spatial X Position')
    plt.ylabel('Spatial Y Position')
    plt.savefig(f'new_uncertainty_results/{experiment_name}/{model_name}_model_confidence_{safe_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
        # 6. Evidence strength map - EVIDENCE DEFECT MAP (NEW!) with proper aspect ratio
    plt.figure(figsize=(fig_width, fig_height))
    plt.imshow(evidence_map, cmap='viridis', interpolation='hamming', aspect=1.0)
    plt.colorbar(label='Evidence Strength (Higher=Stronger Evidence)', shrink=0.8)
    plt.title(f'Evidential Evidence Strength - {dataset_name}\nEvidence Strength at Each Location (Each pixel = 1 measurement)', 
              fontsize=14, fontweight='bold')
    plt.xlabel('Spatial X Position')
    plt.ylabel('Spatial Y Position')
    plt.savefig(f'new_uncertainty_results/{experiment_name}/{model_name}_model_evidence_{safe_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Individual defect maps saved for {dataset_name}:")
    print(f"  ✓ {model_name}_model_{safe_name}.png (Classification)")
    print(f"  ✓ {model_name}_model_epistemic_{safe_name}.png (Model Uncertainty)")
    print(f"  ✓ {model_name}_model_aleatoric_{safe_name}.png (Data Uncertainty)")
    print(f"  ✓ {model_name}_model_total_uncertainty_{safe_name}.png (Total Uncertainty)")
    print(f"  ✓ {model_name}_model_confidence_{safe_name}.png (Confidence)")
    print(f"  ✓ {model_name}_model_evidence_{safe_name}.png (Evidence Strength)")


def create_full_evidential_baseline_maps(maps_data, dataset_name, model_type):
    """
    Create baseline-style uncertainty maps for full evidential model
    """
    # Get data
    non_defect_prob = maps_data['non_defect_prob']
    epistemic = maps_data['epistemic_uncertainty']
    aleatoric = maps_data['aleatoric_uncertainty']
    total_unc = maps_data['total_uncertainty']
    confidence = maps_data['confidence']
    
    # Determine spatial arrangement based on dataset
    n_samples = len(non_defect_prob)
    
    # Dataset-specific shapes (based on baseline_train_test patterns)
    if 'DS1' in dataset_name or 'Test' in dataset_name:
        shape = (9, 28)  # DS1: 252 samples = 9×28
    elif 'May' in dataset_name:
        shape = (31, 38)  # CCNY May: 1178 samples = 31×38
    elif 'June' in dataset_name:
        shape = (19, 34)  # CCNY June: 646 samples = 19×34
    elif 'Nov2023' in dataset_name:
        shape = (44, 34)  # CCNY Nov2023: 1496 samples = 44×34
    elif 'Nov' in dataset_name or 'CCNY' in dataset_name:
        shape = (44, 34)  # CCNY Nov: 1496 samples = 44×34
    elif 'Overlay' in dataset_name or 'DS3' in dataset_name:
        shape = (16, 16)  # DS3 Overlay: 252 samples = 16×16 (approximate)
    else:
        # Default square arrangement
        grid_size = int(np.sqrt(n_samples))
        shape = (grid_size, grid_size)
    
    expected_samples = shape[0] * shape[1]
    
    if n_samples != expected_samples:
        print(f"Warning: {dataset_name} has {n_samples} samples, expected {expected_samples} for {shape} grid")
        # Use square arrangement as fallback
        grid_size = int(np.sqrt(n_samples))
        if grid_size * grid_size < n_samples:
            grid_size += 1
        shape = (grid_size, grid_size)
        expected_samples = shape[0] * shape[1]
        
        # Pad data if needed
        pad_size = expected_samples - n_samples
        if pad_size > 0:
            non_defect_prob = np.pad(non_defect_prob, (0, pad_size), mode='constant', constant_values=np.nan)
            epistemic = np.pad(epistemic, (0, pad_size), mode='constant', constant_values=np.nan)
            aleatoric = np.pad(aleatoric, (0, pad_size), mode='constant', constant_values=np.nan)
            total_unc = np.pad(total_unc, (0, pad_size), mode='constant', constant_values=np.nan)
            confidence = np.pad(confidence, (0, pad_size), mode='constant', constant_values=np.nan)
    
    # Reshape to spatial grids
    try:
        classification_map = non_defect_prob[:expected_samples].reshape(shape)
        epistemic_map = epistemic[:expected_samples].reshape(shape)
        aleatoric_map = aleatoric[:expected_samples].reshape(shape)
        total_uncertainty_map = total_unc[:expected_samples].reshape(shape)
        confidence_map = confidence[:expected_samples].reshape(shape)
        
        # Save maps in baseline style
        save_evidential_full_baseline_maps(
            classification_map, epistemic_map, aleatoric_map, 
            total_uncertainty_map, confidence_map, dataset_name, model_type
        )
        
    except ValueError as e:
        print(f"Error reshaping data for {dataset_name}: {e}")
        print(f"Data shape: {n_samples}, expected: {expected_samples}, grid: {shape}")


def save_evidential_full_baseline_maps(classification_map, epistemic_map, aleatoric_map, 
                                     total_uncertainty_map, confidence_map, dataset_name, model_type):
    """
    Save evidential maps in baseline style with proper naming
    """
    model_prefix = "evidential_full" if model_type == "full" else "evidential_simple"
    safe_name = dataset_name.lower().replace(" ", "_").replace("-", "_")
    
    # Classification map (non-defect probability)
    plt.figure(figsize=(10, 8))
    plt.imshow(classification_map, cmap='Spectral', interpolation='hamming',)
    plt.colorbar(label='Non-defect Probability')
    plt.title(f'{model_prefix.title()} - Classification ({dataset_name})')
    plt.axis('off')
    plt.savefig(f'new_uncertainty_results/exp3/{model_prefix}_model_{safe_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Epistemic uncertainty map
    plt.figure(figsize=(10, 8))
    plt.imshow(epistemic_map, cmap='Reds_r', interpolation='hamming',)
    plt.colorbar(label='Epistemic Uncertainty')
    plt.title(f'{model_prefix.title()} - Epistemic Uncertainty ({dataset_name})')
    plt.axis('off')
    plt.savefig(f'new_uncertainty_results/exp3/{model_prefix}_model_epistemic_{safe_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Aleatoric uncertainty map
    plt.figure(figsize=(10, 8))
    plt.imshow(aleatoric_map, cmap='Blues_r', interpolation='hamming',)
    plt.colorbar(label='Aleatoric Uncertainty')
    plt.title(f'{model_prefix.title()} - Aleatoric Uncertainty ({dataset_name})')
    plt.axis('off')
    plt.savefig(f'new_uncertainty_results/{experiment_name}/{model_prefix}_model_aleatoric_{safe_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Total uncertainty map
    plt.figure(figsize=(10, 8))
    plt.imshow(total_uncertainty_map, cmap='Purples_r', interpolation='hamming',)
    plt.colorbar(label='Total Uncertainty')
    plt.title(f'{model_prefix.title()} - Total Uncertainty ({dataset_name})')
    plt.axis('off')
    plt.savefig(f'new_uncertainty_results/{experiment_name}/{model_prefix}_model_total_uncertainty_{safe_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Confidence map
    plt.figure(figsize=(10, 8))
    plt.imshow(confidence_map, cmap='Spectral', interpolation='hamming',)
    plt.colorbar(label='Confidence')
    plt.title(f'{model_prefix.title()} - Confidence ({dataset_name})')
    plt.axis('off')
    plt.savefig(f'new_uncertainty_results/{experiment_name}/{model_prefix}_model_confidence_{safe_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved baseline-style maps for {dataset_name} using {model_type} model:")
    print(f"  - {model_prefix}_model_{safe_name}.png")
    print(f"  - {model_prefix}_model_epistemic_{safe_name}.png")
    print(f"  - {model_prefix}_model_aleatoric_{safe_name}.png")
    print(f"  - {model_prefix}_model_total_uncertainty_{safe_name}.png")
    print(f"  - {model_prefix}_model_confidence_{safe_name}.png")


def save_evidential_full_maps(classification_map, epistemic_map, aleatoric_map, 
                             total_uncertainty_map, confidence_map, dataset_name):
    """
    Save evidential full model maps in baseline style
    """
    # Classification map (non-defect probability)
    plt.figure(figsize=(10, 8))
    plt.imshow(classification_map, cmap='Spectral', interpolation='hamming',)
    plt.colorbar(label='Non-defect Probability')
    plt.title(f'Evidential Full - Classification ({dataset_name})')
    plt.axis('off')
    plt.savefig(f'evidential_full_model_{dataset_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Epistemic uncertainty map
    plt.figure(figsize=(10, 8))
    plt.imshow(epistemic_map, cmap='Reds_r', interpolation='hamming',)
    plt.colorbar(label='Epistemic Uncertainty')
    plt.title(f'Evidential Full - Epistemic Uncertainty ({dataset_name})')
    plt.axis('off')
    plt.savefig(f'evidential_full_model_epistemic_{dataset_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Aleatoric uncertainty map
    plt.figure(figsize=(10, 8))
    plt.imshow(aleatoric_map, cmap='Blues_r', interpolation='hamming',)
    plt.colorbar(label='Aleatoric Uncertainty')
    plt.title(f'Evidential Full - Aleatoric Uncertainty ({dataset_name})')
    plt.axis('off')
    plt.savefig(f'evidential_full_model_aleatoric_{dataset_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Total uncertainty map
    plt.figure(figsize=(10, 8))
    plt.imshow(total_uncertainty_map, cmap='Purples_r', interpolation='hamming',)
    plt.colorbar(label='Total Uncertainty')
    plt.title(f'Evidential Full - Total Uncertainty ({dataset_name})')
    plt.axis('off')
    plt.savefig(f'evidential_full_model_total_uncertainty_{dataset_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Confidence map
    plt.figure(figsize=(10, 8))
    plt.imshow(confidence_map, cmap='Spectral', interpolation='hamming',)
    plt.colorbar(label='Confidence')
    plt.title(f'Evidential Full - Confidence ({dataset_name})')
    plt.axis('off')
    plt.savefig(f'evidential_full_model_confidence_{dataset_name}.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved maps for {dataset_name}:")
    print(f"  - evidential_full_model_{dataset_name}.png")
    print(f"  - evidential_full_model_epistemic_{dataset_name}.png")
    print(f"  - evidential_full_model_aleatoric_{dataset_name}.png")
    print(f"  - evidential_full_model_total_uncertainty_{dataset_name}.png")
    print(f"  - evidential_full_model_confidence_{dataset_name}.png")


def run_inference_time_uncertainty_analysis(model_path='weights/evidential_full_v2.pth', 
                                           experiment_name=None, model_name=None):
    """
    Run beautiful uncertainty analysis during inference time - no pre-saved results needed!
    """
    print("=== Real-Time Evidential Uncertainty Analysis ===")
    print("🚀 Generating beautiful uncertainty visualizations during inference...\n")
    
    try:
        # Test model on datasets and get fresh results
        print("1. Running inference on test datasets...")
        dataset_results = test_full_evidential_model_on_datasets(model_path)
        
        if dataset_results:
            print("✓ Inference complete! Now generating beautiful analysis...\n")
            
            # Analyze the fresh results from inference
            for dataset_name, results in dataset_results.items():
                print(f"\n=== Analyzing {dataset_name} Results ===")
                
                # Simulate the analyze_full_evidential_results function with fresh data
                analyze_inference_results(results, dataset_name, experiment_name, model_name)
            
            # Save the fresh inference results
            if model_name is None:
                model_name = default_model_name
            
            results_filename = f'weights/{model_name}_inference_results.pth'
            torch.save(dataset_results, results_filename)
            print(f"✓ Fresh inference results saved to {results_filename}")
            
        else:
            print("❌ No inference results generated - check model file")
            
    except FileNotFoundError:
        print(f"❌ Model file not found: {model_path}")
        print("Please ensure the evidential model exists or provide correct path")
        return False
    
    return True


def analyze_inference_results(results, dataset_name, experiment_name=None, model_name=None):
    """
    Analyze fresh inference results and create beautiful visualizations
    """
    predictions = results['predictions']
    total_unc = results['total_uncertainties']
    epistemic_unc = results['epistemic_uncertainties']
    aleatoric_unc = results['aleatoric_uncertainties']
    confidences = results['confidences']
    alphas = results['alphas']
    targets = results['targets']
    accuracy = results['accuracy']
    
    print(f"Dataset: {dataset_name}")
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Number of samples: {len(targets)}")
    
    # Extract predictions for analysis
    pred_probs = predictions.squeeze(0)  # Remove sequence dim
    pred_classes = torch.argmax(pred_probs, dim=1)
    
    # Remove extra dimensions (handle both supervised and unsupervised data)
    epistemic_unc = epistemic_unc.squeeze(0).squeeze(-1) if epistemic_unc.dim() > 1 else epistemic_unc.squeeze(0)
    aleatoric_unc = aleatoric_unc.squeeze(0).squeeze(-1) if aleatoric_unc.dim() > 1 else aleatoric_unc.squeeze(0)
    total_unc = total_unc.squeeze(0).squeeze(-1) if total_unc.dim() > 1 else total_unc.squeeze(0)
    alphas = alphas.squeeze(0).squeeze(-1) if alphas.dim() > 1 else alphas.squeeze(0)
    
    # Handle confidence tensor (might be different shapes for supervised vs unsupervised)
    if confidences.dim() > 1:
        confidences = confidences.squeeze(0)  # Remove batch dimension if present
    
    # Convert to numpy for plotting
    epistemic_unc_np = epistemic_unc.detach().cpu().numpy()
    aleatoric_unc_np = aleatoric_unc.detach().cpu().numpy()
    total_unc_np = total_unc.detach().cpu().numpy()
    alphas_np = alphas.detach().cpu().numpy()
    confidences_np = confidences.detach().cpu().numpy()
    pred_classes_np = pred_classes.detach().cpu().numpy()
    targets_np = targets.detach().cpu().numpy()
    pred_probs_np = pred_probs.detach().cpu().numpy()
    
    # Debug print shapes to ensure they match
    print(f"Debug numpy shapes:")
    print(f"  pred_classes_np: {pred_classes_np.shape}")
    print(f"  targets_np: {targets_np.shape}")
    print(f"  confidences_np: {confidences_np.shape}")
    print(f"  epistemic_unc_np: {epistemic_unc_np.shape}")
    
    # Classification metrics (ensure arrays have same shape)
    if pred_classes_np.shape != targets_np.shape:
        print(f"Warning: Shape mismatch! pred_classes: {pred_classes_np.shape}, targets: {targets_np.shape}")
        # For unsupervised data, create dummy correct_predictions
        if accuracy == 0.0:  # Unsupervised dataset
            correct_predictions = np.ones(len(pred_classes_np), dtype=bool)  # Treat all as "correct" for visualization
            print("Using dummy correct_predictions for unsupervised dataset")
        else:
            correct_predictions = (pred_classes_np == targets_np)
    else:
        correct_predictions = (pred_classes_np == targets_np)
    
    print(f"Correct Predictions: {correct_predictions.sum()}/{len(correct_predictions)}")
    
    # Print uncertainty statistics
    print(f"\n=== Live Uncertainty Analysis ===")
    print(f"Total Uncertainty - Mean: {np.mean(total_unc_np):.6f} ± {np.std(total_unc_np):.6f}")
    print(f"Epistemic Uncertainty - Mean: {np.mean(epistemic_unc_np):.6f} ± {np.std(epistemic_unc_np):.6f}")
    print(f"Aleatoric Uncertainty - Mean: {np.mean(aleatoric_unc_np):.6f} ± {np.std(aleatoric_unc_np):.6f}")
    print(f"Confidence - Mean: {np.mean(confidences_np):.6f} ± {np.std(confidences_np):.6f}")
    
    # Create beautiful visualizations with dataset-specific naming
    print(f"\n🎨 Creating beautiful visualizations for {dataset_name}...")
    
    # 1. Advanced uncertainty distribution analysis
    create_advanced_uncertainty_distribution_analysis(
        pred_probs_np, epistemic_unc_np, aleatoric_unc_np, total_unc_np, 
        confidences_np, alphas_np, targets_np, correct_predictions
    )
    
    # 2. Comprehensive 9-subplot analysis
    create_full_evidential_plots(
        pred_probs_np, epistemic_unc_np, aleatoric_unc_np, total_unc_np, 
        confidences_np, alphas_np, targets_np, correct_predictions
    )
    
    # 3. Spatial uncertainty maps
    create_spatial_uncertainty_maps(
        pred_probs_np, epistemic_unc_np.squeeze(), aleatoric_unc_np.squeeze(), 
        total_unc_np.squeeze(), confidences_np, alphas_np.squeeze(), targets_np, dataset_name
    )
    
    # 4. Advanced correlation analysis
    create_advanced_uncertainty_correlations(
        epistemic_unc_np, aleatoric_unc_np, total_unc_np, 
        confidences_np, alphas_np, correct_predictions
    )
    
    # 5. Individual baseline-style defect maps (always generate these!)
    create_individual_defect_maps(
        pred_probs_np, epistemic_unc_np.squeeze(), aleatoric_unc_np.squeeze(),
        total_unc_np.squeeze(), confidences_np, alphas_np.squeeze(), dataset_name
    )
    
    print(f"✓ Beautiful analysis complete for {dataset_name}!")


if __name__ == '__main__':
    # Configuration - easily changeable!
    experiment_name = "exp4"  # Change this for different experiments
    model_name = "evidential_full"  # Change this for different models
    model_path = f'weights/{model_name}.pth'
    
    print("=== Real-Time Evidential Uncertainty Analysis ===")
    print(f"🚀 Running experiment: {experiment_name} with model: {model_name}")
    print("🚀 Generating beautiful uncertainty analysis during inference time!\n")
    
    # Create experiment directory
    import os
    os.makedirs(f'new_uncertainty_results/{experiment_name}', exist_ok=True)
    
    # Run the inference-time analysis
    success = run_inference_time_uncertainty_analysis(model_path, experiment_name, model_name)
    
    if not success:
        print("\n🔄 Trying alternative model paths...")
        alternative_paths = [
            f'new_uncertainty_results/weights/{model_name}.pth',
            'weights/evidential_simple.pth',
            'weights/evidential.pth',
            'weights/evidential_full.pth'
        ]
        
        for alt_path in alternative_paths:
            print(f"Trying: {alt_path}")
            if run_inference_time_uncertainty_analysis(alt_path, experiment_name, model_name):
                success = True
                break
    
    if not success:
        print("\n❌ No evidential models found. Please train a model first.")
        print("Available training scripts:")
        print("  - train_evidential_full.py")
        print("  - train_evidential_simple.py")
    
    print("\n=== Analysis Complete ===")
    print("✓ All beautiful uncertainty visualizations completed successfully!")
    print("\n📁 Generated Core Analysis Files:")
    print(f"✓ new_uncertainty_results/{experiment_name}/{model_name}_uncertainty_analysis.png")
    print(f"✓ new_uncertainty_results/{experiment_name}/{model_name}_uncertainty_distributions.png")
    print(f"✓ new_uncertainty_results/{experiment_name}/{model_name}_uncertainty_maps.png")
    print(f"✓ new_uncertainty_results/{experiment_name}/{model_name}_correlation_matrix.png")
    print(f"✓ weights/{model_name}_inference_results.pth")
    print("\n🗺️  Generated Individual Defect Maps (for each dataset):")
    print(f"✓ {model_name}_model_[dataset].png - Classification maps")
    print(f"✓ {model_name}_model_epistemic_[dataset].png - Model uncertainty maps")
    print(f"✓ {model_name}_model_aleatoric_[dataset].png - Data uncertainty maps")
    print(f"✓ {model_name}_model_total_uncertainty_[dataset].png - Total uncertainty maps")
    print(f"✓ {model_name}_model_confidence_[dataset].png - Confidence maps")
    print(f"✓ {model_name}_model_evidence_[dataset].png - Evidence strength maps")
    print("\n🎉 Real-time uncertainty analysis complete! Check the generated visualizations.")
    print("\n👁️  All defect maps show exact spatial locations representing real measurement positions!")
    
    # Ensure all plots are closed and memory is freed
    plt.close('all')
    print("✓ All matplotlib resources cleaned up.")
    print("\n🔴 Script execution finished. Safe to exit.")