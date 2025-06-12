import warnings
warnings.filterwarnings("ignore")
import sys
import os
sys.path.insert(0, '/Users/evhoxha/projects/impact_echo_contrastive_learning/')

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

def load_and_analyze_evidential_results(results_path='weights/evidential_model_results.pth'):
    """
    Load and analyze evidential results using only numpy operations
    """
    print("Loading evidential results...")
    
    # Try to import torch here to avoid global import issues
    try:
        import torch
        results = torch.load(results_path, map_location='cpu')
        
        # Convert everything to numpy immediately
        predictions = results['predictions'].detach().cpu().numpy()
        epistemic_unc = results['epistemic_uncertainty'].detach().cpu().numpy()
        aleatoric_unc = results['aleatoric_uncertainty'].detach().cpu().numpy()
        total_unc = results['total_uncertainty'].detach().cpu().numpy()
        confidences = results['confidences'].detach().cpu().numpy()
        alpha_sums = results['alpha_sums'].detach().cpu().numpy()
        targets = results['targets'].detach().cpu().numpy()
        test_accuracy = results['test_accuracy']
        
        print(f"Test Accuracy: {test_accuracy:.2f}%")
        print(f"Number of test samples: {len(targets)}")
        
        return {
            'predictions': predictions,
            'epistemic_uncertainty': epistemic_unc,
            'aleatoric_uncertainty': aleatoric_unc,
            'total_uncertainty': total_unc,
            'confidences': confidences,
            'alpha_sums': alpha_sums,
            'targets': targets,
            'test_accuracy': test_accuracy
        }
    except Exception as e:
        print(f"Error loading results: {e}")
        return None

def create_baseline_style_maps(results, pred_grid, epistemic_grid, aleatoric_grid, 
                              total_unc_grid, confidence_grid):
    """
    Create individual maps in the same style as other baseline methods
    """
    # Use predictions to get class 0 probabilities (non-defect) - matching baseline approach
    predictions = results['predictions']
    if len(predictions.shape) > 1 and predictions.shape[1] > 1:
        # Get class 0 probabilities (non-defect) - matching baseline approach
        class0_probs = predictions[:, 0]
    else:
        # Binary predictions, convert to probabilities
        class0_probs = 1 - predictions.squeeze()
    
    # Use the same shape as pred_grid
    shape = pred_grid.shape
    
    # Reshape class 0 probabilities to match spatial dimensions
    if len(class0_probs) == 252:
        # DS1 dataset - reshape to (9, 28) like baseline methods
        class0_grid = class0_probs.reshape(9, 28)
    else:
        # For other datasets, use the shape determined in the main function
        if len(class0_probs) == shape[0] * shape[1]:
            class0_grid = class0_probs.reshape(shape)
        else:
            # Pad if necessary
            pad_size = shape[0] * shape[1] - len(class0_probs)
            if pad_size > 0:
                class0_probs = np.pad(class0_probs, (0, pad_size), mode='constant', constant_values=np.nan)
            class0_grid = class0_probs.reshape(shape)
    
    # Create individual maps matching baseline style
    
    # 1. Classification map (showing non-defect probability, matching baseline)
    plt.figure(figsize=(8, 6))
    im = plt.imshow(class0_grid, cmap='Spectral', interpolation='hamming')
    plt.colorbar(im, label='Non-defect Probability')
    plt.axis("OFF")
    plt.savefig('evidential_model_ds1.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Evidential DS1 - Classification Map Generated")
    
    # 2. Epistemic uncertainty map (reversed colormap for higher uncertainty = darker)
    plt.figure(figsize=(8, 6))
    im = plt.imshow(epistemic_grid, cmap='Reds_r', interpolation='hamming')
    plt.colorbar(im, label='Epistemic Uncertainty')
    plt.axis("OFF")
    plt.savefig('evidential_model_epistemic_ds1.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Evidential DS1 - Epistemic Uncertainty Map Generated")
    
    # 3. Aleatoric uncertainty map (reversed colormap for higher uncertainty = darker)
    plt.figure(figsize=(8, 6))
    im = plt.imshow(aleatoric_grid, cmap='Blues_r', interpolation='hamming')
    plt.colorbar(im, label='Aleatoric Uncertainty')
    plt.axis("OFF")
    plt.savefig('evidential_model_aleatoric_ds1.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Evidential DS1 - Aleatoric Uncertainty Map Generated")
    
    # 4. Total uncertainty map (reversed colormap for higher uncertainty = darker)
    plt.figure(figsize=(8, 6))
    im = plt.imshow(total_unc_grid, cmap='Purples_r', interpolation='hamming')
    plt.colorbar(im, label='Total Uncertainty')
    plt.axis("OFF")
    plt.savefig('evidential_model_total_uncertainty_ds1.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Evidential DS1 - Total Uncertainty Map Generated")
    
    # 5. Confidence map (keep same as baseline)
    plt.figure(figsize=(8, 6))
    im = plt.imshow(confidence_grid, cmap='Spectral', interpolation='hamming')
    plt.colorbar(im, label='Confidence')
    plt.axis("OFF")
    plt.savefig('evidential_model_confidence_ds1.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Evidential DS1 - Confidence Map Generated")

def create_uncertainty_maps(results):
    """
    Create comprehensive uncertainty maps from results
    """
    if results is None:
        print("No results to process")
        return
    
    # Extract data
    predictions = results['predictions'].squeeze()
    epistemic = results['epistemic_uncertainty'].squeeze()
    aleatoric = results['aleatoric_uncertainty'].squeeze()
    total_unc = results['total_uncertainty'].squeeze()
    confidences = results['confidences'].squeeze()
    alpha_sums = results['alpha_sums'].squeeze()
    targets = results['targets']
    
    # Get predicted classes
    if len(predictions.shape) > 1:
        pred_classes = np.argmax(predictions, axis=1)
    else:
        pred_classes = predictions
    
    # Calculate accuracy metrics
    correct_predictions = (pred_classes == targets)
    incorrect_predictions = ~correct_predictions
    
    print(f"\nUncertainty Analysis:")
    print(f"Correct Predictions: {np.sum(correct_predictions)}/{len(targets)}")
    print(f"Incorrect Predictions: {np.sum(incorrect_predictions)}/{len(targets)}")
    
    print(f"Correct - Mean Total Uncertainty: {np.mean(total_unc[correct_predictions]):.4f}")
    print(f"Incorrect - Mean Total Uncertainty: {np.mean(total_unc[incorrect_predictions]):.4f}")
    
    print(f"Correct - Mean Epistemic Uncertainty: {np.mean(epistemic[correct_predictions]):.4f}")
    print(f"Incorrect - Mean Epistemic Uncertainty: {np.mean(epistemic[incorrect_predictions]):.4f}")
    
    print(f"Correct - Mean Confidence: {np.mean(confidences[correct_predictions]):.4f}")
    print(f"Incorrect - Mean Confidence: {np.mean(confidences[incorrect_predictions]):.4f}")
    
    # DS1 test data should be reshaped to (9, 28) to match baseline methods
    # Total samples: 252 = 9 × 28
    n_samples = len(pred_classes)
    print(f"Number of samples: {n_samples}")
    
    if n_samples == 252:
        # DS1 dataset - reshape to (9, 28) like baseline methods
        shape = (9, 28)
        pred_grid = pred_classes.reshape(shape)
        epistemic_grid = epistemic.reshape(shape)
        aleatoric_grid = aleatoric.reshape(shape)
        total_unc_grid = total_unc.reshape(shape)
        confidence_grid = confidences.reshape(shape)
        evidence_grid = alpha_sums.reshape(shape)
        target_grid = targets.reshape(shape)
    else:
        # For other datasets, use square grid as fallback
        grid_size = int(np.sqrt(n_samples))
        if grid_size * grid_size < n_samples:
            grid_size += 1
        
        # Pad data to fit grid
        pad_size = grid_size * grid_size - n_samples
        if pad_size > 0:
            pred_classes = np.pad(pred_classes, (0, pad_size), mode='constant', constant_values=-1)
            epistemic = np.pad(epistemic, (0, pad_size), mode='constant', constant_values=np.nan)
            aleatoric = np.pad(aleatoric, (0, pad_size), mode='constant', constant_values=np.nan)
            total_unc = np.pad(total_unc, (0, pad_size), mode='constant', constant_values=np.nan)
            confidences = np.pad(confidences, (0, pad_size), mode='constant', constant_values=np.nan)
            alpha_sums = np.pad(alpha_sums, (0, pad_size), mode='constant', constant_values=np.nan)
            targets = np.pad(targets, (0, pad_size), mode='constant', constant_values=-1)
        
        # Reshape to 2D grids
        shape = (grid_size, grid_size)
        pred_grid = pred_classes.reshape(shape)
        epistemic_grid = epistemic.reshape(shape)
        aleatoric_grid = aleatoric.reshape(shape)
        total_unc_grid = total_unc.reshape(shape)
        confidence_grid = confidences.reshape(shape)
        evidence_grid = alpha_sums.reshape(shape)
        target_grid = targets.reshape(shape)
    
    # Create individual baseline-style maps (like other methods)
    create_baseline_style_maps(results, pred_grid, epistemic_grid, aleatoric_grid, 
                               total_unc_grid, confidence_grid)
    
    print(f"Shape used for grids: {pred_grid.shape}")
    
    # Create comprehensive uncertainty visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Evidential Deep Learning Uncertainty Maps', fontsize=16)
    
    # Custom colormaps
    uncertainty_cmap = LinearSegmentedColormap.from_list('uncertainty', ['white', 'yellow', 'red'])
    confidence_cmap = LinearSegmentedColormap.from_list('confidence', ['red', 'yellow', 'green'])
    
    # Plot 1: Predictions
    im1 = axes[0, 0].imshow(pred_grid, cmap='RdYlBu_r', interpolation='nearest')
    axes[0, 0].set_title('Predictions (0=No Defect, 1=Defect)')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Plot 2: Epistemic uncertainty
    im2 = axes[0, 1].imshow(epistemic_grid, cmap=uncertainty_cmap, interpolation='nearest')
    axes[0, 1].set_title('Epistemic Uncertainty (Model Uncertainty)')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Plot 3: Aleatoric uncertainty
    im3 = axes[0, 2].imshow(aleatoric_grid, cmap=uncertainty_cmap, interpolation='nearest')
    axes[0, 2].set_title('Aleatoric Uncertainty (Data Uncertainty)')
    plt.colorbar(im3, ax=axes[0, 2])
    
    # Plot 4: Total uncertainty
    im4 = axes[1, 0].imshow(total_unc_grid, cmap=uncertainty_cmap, interpolation='nearest')
    axes[1, 0].set_title('Total Uncertainty')
    plt.colorbar(im4, ax=axes[1, 0])
    
    # Plot 5: Confidence
    im5 = axes[1, 1].imshow(confidence_grid, cmap=confidence_cmap, interpolation='nearest')
    axes[1, 1].set_title('Prediction Confidence')
    plt.colorbar(im5, ax=axes[1, 1])
    
    # Plot 6: Evidence strength
    im6 = axes[1, 2].imshow(evidence_grid, cmap='viridis', interpolation='nearest')
    axes[1, 2].set_title('Evidence Strength (Alpha Sum)')
    plt.colorbar(im6, ax=axes[1, 2])
    
    # Remove axis ticks
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig('weights/evidential_uncertainty_maps.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Uncertainty maps saved to weights/evidential_uncertainty_maps.png")
    
    # Create defect detection maps
    create_defect_maps(pred_grid, total_unc_grid, confidence_grid, target_grid)

def create_defect_maps(pred_grid, total_unc_grid, confidence_grid, target_grid):
    """
    Create defect detection maps with uncertainty overlay
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle('Defect Detection with Uncertainty Analysis', fontsize=16)
    
    # Ground truth
    im1 = axes[0, 0].imshow(target_grid, cmap='RdYlBu_r', interpolation='nearest')
    axes[0, 0].set_title('Ground Truth')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Predictions
    im2 = axes[0, 1].imshow(pred_grid, cmap='RdYlBu_r', interpolation='nearest')
    axes[0, 1].set_title('Predictions')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Predictions + Uncertainty overlay
    axes[1, 0].imshow(pred_grid, cmap='RdYlBu_r', alpha=0.7, interpolation='nearest')
    im3 = axes[1, 0].imshow(total_unc_grid, cmap='Reds', alpha=0.5, interpolation='nearest')
    axes[1, 0].set_title('Predictions + Uncertainty Overlay')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # High uncertainty regions
    high_unc_threshold = np.nanmean(total_unc_grid) + 2 * np.nanstd(total_unc_grid)
    high_unc_mask = total_unc_grid > high_unc_threshold
    axes[1, 1].imshow(pred_grid, cmap='RdYlBu_r', alpha=0.7, interpolation='nearest')
    axes[1, 1].imshow(high_unc_mask, cmap='Reds', alpha=0.8, interpolation='nearest')
    axes[1, 1].set_title('High Uncertainty Regions (>μ+2σ)')
    
    # Remove axis ticks
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    plt.savefig('weights/evidential_defect_maps.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Defect detection maps saved to weights/evidential_defect_maps.png")

def create_uncertainty_distribution_plots(results):
    """
    Create uncertainty distribution analysis plots
    """
    if results is None:
        return
    
    epistemic = results['epistemic_uncertainty'].squeeze()
    aleatoric = results['aleatoric_uncertainty'].squeeze()
    total_unc = results['total_uncertainty'].squeeze()
    confidences = results['confidences'].squeeze()
    predictions = results['predictions'].squeeze()
    targets = results['targets']
    
    if len(predictions.shape) > 1:
        pred_classes = np.argmax(predictions, axis=1)
    else:
        pred_classes = predictions
    
    correct_predictions = (pred_classes == targets)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('Evidential Uncertainty Distribution Analysis', fontsize=16)
    
    # Epistemic vs Aleatoric scatter
    axes[0, 0].scatter(epistemic[correct_predictions], aleatoric[correct_predictions], 
                      alpha=0.6, c='green', label='Correct', s=30)
    axes[0, 0].scatter(epistemic[~correct_predictions], aleatoric[~correct_predictions], 
                      alpha=0.6, c='red', label='Incorrect', s=30)
    axes[0, 0].set_xlabel('Epistemic Uncertainty')
    axes[0, 0].set_ylabel('Aleatoric Uncertainty')
    axes[0, 0].set_title('Epistemic vs Aleatoric Uncertainty')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Total uncertainty distribution
    axes[0, 1].hist(total_unc[correct_predictions], bins=30, alpha=0.7, 
                   color='green', label='Correct', density=True)
    axes[0, 1].hist(total_unc[~correct_predictions], bins=30, alpha=0.7, 
                   color='red', label='Incorrect', density=True)
    axes[0, 1].set_xlabel('Total Uncertainty')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('Total Uncertainty Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Confidence distribution
    axes[1, 0].hist(confidences[correct_predictions], bins=30, alpha=0.7, 
                   color='green', label='Correct', density=True)
    axes[1, 0].hist(confidences[~correct_predictions], bins=30, alpha=0.7, 
                   color='red', label='Incorrect', density=True)
    axes[1, 0].set_xlabel('Confidence')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Confidence Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Uncertainty vs Confidence
    axes[1, 1].scatter(total_unc[correct_predictions], confidences[correct_predictions], 
                      alpha=0.6, c='green', label='Correct', s=30)
    axes[1, 1].scatter(total_unc[~correct_predictions], confidences[~correct_predictions], 
                      alpha=0.6, c='red', label='Incorrect', s=30)
    axes[1, 1].set_xlabel('Total Uncertainty')
    axes[1, 1].set_ylabel('Confidence')
    axes[1, 1].set_title('Uncertainty vs Confidence')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('weights/evidential_uncertainty_distributions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Uncertainty distribution plots saved to weights/evidential_uncertainty_distributions.png")

if __name__ == '__main__':
    print("=== Simple Evidential Deep Learning Mapping ===\n")
    
    # Load and analyze results
    results = load_and_analyze_evidential_results('weights/evidential_model_results.pth')
    
    if results is not None:
        print("\nGenerating uncertainty maps...")
        create_uncertainty_maps(results)
        
        print("\nGenerating distribution analysis...")
        create_uncertainty_distribution_plots(results)
        
        print("\n=== Mapping Complete ===")
        print("Generated files:")
        print("  - weights/evidential_uncertainty_maps.png")
        print("  - weights/evidential_defect_maps.png")
        print("  - weights/evidential_uncertainty_distributions.png")
    else:
        print("Could not load evidential results.")
        print("Please ensure the evidential model has been trained and results saved.")