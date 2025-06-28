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
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from utils import *

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

from models.model_parts import ResidualBlock
from train_evidential import EvidentialIENet


def analyze_evidential_results(results_path):
    """
    Analyze and visualize evidential deep learning results
    """
    print(f"Loading results from {results_path}")
    results = torch.load(results_path, map_location='cpu')
    
    predictions = results['predictions']
    epistemic_unc = results['epistemic_uncertainty']
    aleatoric_unc = results['aleatoric_uncertainty']
    total_unc = results['total_uncertainty']
    confidences = results['confidences']
    alpha_sums = results['alpha_sums']
    targets = results['targets']
    test_accuracy = results['test_accuracy']
    
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    print(f"Number of test samples: {len(targets)}")
    
    # Extract predictions for analysis
    pred_probs = predictions.squeeze(0)  # Remove sequence dim
    pred_classes = torch.argmax(pred_probs, dim=1)
    
    epistemic_unc = epistemic_unc.squeeze(0).squeeze(1)  # Remove extra dims
    aleatoric_unc = aleatoric_unc.squeeze(0).squeeze(1)
    total_unc = total_unc.squeeze(0).squeeze(1)
    alpha_sums = alpha_sums.squeeze(0).squeeze(1)
    
    # Convert to numpy for plotting
    epistemic_unc = epistemic_unc.detach().cpu().numpy()
    aleatoric_unc = aleatoric_unc.detach().cpu().numpy()
    total_unc = total_unc.detach().cpu().numpy()
    alpha_sums = alpha_sums.detach().cpu().numpy()
    confidences = confidences.detach().cpu().numpy()
    pred_classes = pred_classes.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()
    
    # Classification metrics
    correct_predictions = (pred_classes == targets)
    incorrect_predictions = ~correct_predictions
    
    print(f"\nCorrect Predictions: {correct_predictions.sum()}/{len(targets)}")
    print(f"Incorrect Predictions: {incorrect_predictions.sum()}/{len(targets)}")
    
    # Uncertainty statistics for correct vs incorrect predictions
    print(f"\nUncertainty Analysis:")
    print(f"Correct Predictions - Mean Total Uncertainty: {np.mean(total_unc[correct_predictions]):.4f}")
    print(f"Incorrect Predictions - Mean Total Uncertainty: {np.mean(total_unc[incorrect_predictions]):.4f}")
    
    print(f"Correct Predictions - Mean Epistemic Uncertainty: {np.mean(epistemic_unc[correct_predictions]):.4f}")
    print(f"Incorrect Predictions - Mean Epistemic Uncertainty: {np.mean(epistemic_unc[incorrect_predictions]):.4f}")
    
    print(f"Correct Predictions - Mean Confidence: {np.mean(confidences[correct_predictions]):.4f}")
    print(f"Incorrect Predictions - Mean Confidence: {np.mean(confidences[incorrect_predictions]):.4f}")
    
    print(f"Correct Predictions - Mean Evidence Strength: {np.mean(alpha_sums[correct_predictions]):.4f}")
    print(f"Incorrect Predictions - Mean Evidence Strength: {np.mean(alpha_sums[incorrect_predictions]):.4f}")
    
    # Identify high uncertainty samples
    high_epistemic_threshold = np.mean(epistemic_unc) + 2 * np.std(epistemic_unc)
    high_total_threshold = np.mean(total_unc) + 2 * np.std(total_unc)
    low_confidence_threshold = np.mean(confidences) - 2 * np.std(confidences)
    
    high_epistemic_samples = epistemic_unc > high_epistemic_threshold
    high_total_samples = total_unc > high_total_threshold
    low_confidence_samples = confidences < low_confidence_threshold
    
    print(f"\nHigh Uncertainty Samples:")
    print(f"High Epistemic Uncertainty (>μ+2σ): {np.sum(high_epistemic_samples)}")
    print(f"High Total Uncertainty (>μ+2σ): {np.sum(high_total_samples)}")
    print(f"Low Confidence (<μ-2σ): {np.sum(low_confidence_samples)}")
    
    # Create visualizations
    create_uncertainty_plots(
        predictions, epistemic_unc, aleatoric_unc, total_unc, 
        confidences, alpha_sums, targets, correct_predictions
    )
    
    return results


def create_uncertainty_plots(predictions, epistemic_unc, aleatoric_unc, total_unc, 
                           confidences, alpha_sums, targets, correct_predictions):
    """
    Create comprehensive uncertainty visualization plots
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Evidential Deep Learning Uncertainty Analysis', fontsize=16)
    
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
    axes[0, 2].scatter(alpha_sums[correct_predictions], confidences[correct_predictions], 
                      alpha=0.6, c='green', label='Correct', s=30)
    axes[0, 2].scatter(alpha_sums[~correct_predictions], confidences[~correct_predictions], 
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
    
    # Plot 5: Confidence Distribution
    axes[1, 1].hist(confidences[correct_predictions], bins=30, alpha=0.7, 
                   color='green', label='Correct', density=True)
    axes[1, 1].hist(confidences[~correct_predictions], bins=30, alpha=0.7, 
                   color='red', label='Incorrect', density=True)
    axes[1, 1].set_xlabel('Confidence')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('Confidence Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Plot 6: Evidence Strength Distribution
    axes[1, 2].hist(alpha_sums[correct_predictions], bins=30, alpha=0.7, 
                   color='green', label='Correct', density=True)
    axes[1, 2].hist(alpha_sums[~correct_predictions], bins=30, alpha=0.7, 
                   color='red', label='Incorrect', density=True)
    axes[1, 2].set_xlabel('Evidence Strength (Alpha Sum)')
    axes[1, 2].set_ylabel('Density')
    axes[1, 2].set_title('Evidence Strength Distribution')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('weights/evidential_uncertainty_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Uncertainty analysis plots saved to weights/evidential_uncertainty_analysis.png")


def test_evidential_model_on_new_data(model_path, test_data_path):
    """
    Test the trained evidential model on new data
    """
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Load model
    model = EvidentialIENet(num_classes=2, verbose=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    
    # Load test data
    X_test_path = [test_data_path]
    y_test_path = ['data/y_test.npy']  # Assuming same labels
    test_dataset = ImpactEchoDatasetClassifier(X_test_path, y_path=y_test_path, array_size=860)
    test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False, num_workers=2)
    
    print(f"Testing model on {len(test_dataset)} samples...")
    
    all_predictions = []
    all_uncertainties = []
    all_confidences = []
    all_targets = []
    
    with torch.no_grad():
        for data in test_loader:
            X = data[0].to(device, dtype=torch.float)
            labels = data[1].to(device, dtype=torch.int).long()
            X = X.view(X.size(0), 1, X.size(1))
            
            prob, epistemic, aleatoric, total_unc, confidence, alpha_sum = model.predict_with_uncertainty(X)
            
            all_predictions.append(prob.cpu())
            all_uncertainties.append(total_unc.cpu())
            all_confidences.append(confidence.cpu())
            all_targets.append(labels.cpu())
    
    predictions = torch.cat(all_predictions, dim=1)
    uncertainties = torch.cat(all_uncertainties, dim=1)
    confidences = torch.cat(all_confidences, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    # Calculate accuracy
    pred_classes = torch.argmax(predictions.squeeze(0), dim=1)
    accuracy = (pred_classes == targets).float().mean() * 100
    
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Mean Uncertainty: {uncertainties.mean():.4f}")
    print(f"Mean Confidence: {confidences.mean():.4f}")
    
    return predictions, uncertainties, confidences, targets


def generate_evidential_uncertainty_maps(model_path='weights/evidential_model.pth', save_maps=True):
    """
    Generate comprehensive uncertainty maps for the evidential method
    """
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load trained model
    model = EvidentialIENet(num_classes=2, verbose=False).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"Loaded evidential model from {model_path}")
    
    # Test on different datasets
    datasets = {
        'Test Set': ('data/X_test_860.npy', 'data/y_test.npy'),
        'CCNY Data': ('data/X_our_slab_size860.npy', None),
        'Overlay Data': ('data/X_overlayed_860.npy', 'data/y_overlayed.npy')
    }
    
    all_maps = {}
    
    for dataset_name, (X_path, y_path) in datasets.items():
        print(f"\nProcessing {dataset_name}...")
        
        try:
            # Load data
            if dataset_name == 'CCNY Data':
                X_may, X_june = load_ccny_sep2022_data_into_torch_tensor(device, X_path)
                maps = process_ccny_data_for_maps(model, X_may, X_june, dataset_name, save_maps)
            elif dataset_name == 'Overlay Data':
                X_overlay, y_overlay = load_ds3_overlay_test_data_into_torch_tensor(device)
                maps = process_dataset_for_maps(model, X_overlay, y_overlay, dataset_name, save_maps)
            else:
                X_test, y_test = load_ds1_test_data_into_torch_tensor(device, X_path, y_path)
                maps = process_dataset_for_maps(model, X_test, y_test, dataset_name, save_maps)
                
            all_maps[dataset_name] = maps
            
        except FileNotFoundError as e:
            print(f"Data file not found for {dataset_name}: {e}")
            continue
    
    return all_maps


def process_dataset_for_maps(model, X_data, y_data, dataset_name, save_maps=True):
    """
    Process a dataset and generate uncertainty maps
    """
    print(f"Processing {len(X_data)} samples for {dataset_name}")
    
    # Get predictions and uncertainties
    with torch.no_grad():
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
    
    maps_data = {
        'predictions': predictions,
        'probabilities': prob_np,
        'epistemic_uncertainty': epistemic_np,
        'aleatoric_uncertainty': aleatoric_np,
        'total_uncertainty': total_unc_np,
        'confidence': confidence_np,
        'evidence_strength': alpha_sum_np,
        'targets': y_data if y_data is not None else None
    }
    
    if save_maps:
        create_uncertainty_heatmaps(maps_data, dataset_name)
        create_defect_detection_maps(maps_data, dataset_name)
    
    return maps_data


def process_ccny_data_for_maps(model, X_may, X_june, dataset_name, save_maps=True):
    """
    Process CCNY data (May and June separately) for uncertainty maps
    """
    maps_data = {}
    
    for period, X_data in [('May', X_may), ('June', X_june)]:
        print(f"Processing {len(X_data)} samples for {dataset_name} - {period}")
        
        with torch.no_grad():
            prob, epistemic, aleatoric, total_unc, confidence, alpha_sum = model.predict_with_uncertainty(X_data)
        
        # Convert to numpy
        prob_np = prob.squeeze(0).cpu().numpy()
        epistemic_np = epistemic.squeeze(0).squeeze(1).cpu().numpy()
        aleatoric_np = aleatoric.squeeze(0).squeeze(1).cpu().numpy()
        total_unc_np = total_unc.squeeze(0).squeeze(1).cpu().numpy()
        confidence_np = confidence.cpu().numpy()
        alpha_sum_np = alpha_sum.squeeze(0).squeeze(1).cpu().numpy()
        
        predictions = np.argmax(prob_np, axis=1)
        
        period_data = {
            'predictions': predictions,
            'probabilities': prob_np,
            'epistemic_uncertainty': epistemic_np,
            'aleatoric_uncertainty': aleatoric_np,
            'total_uncertainty': total_unc_np,
            'confidence': confidence_np,
            'evidence_strength': alpha_sum_np,
            'targets': None
        }
        
        maps_data[period] = period_data
        
        if save_maps:
            create_uncertainty_heatmaps(period_data, f"{dataset_name}_{period}")
            create_defect_detection_maps(period_data, f"{dataset_name}_{period}")
    
    return maps_data


def create_uncertainty_heatmaps(maps_data, dataset_name):
    """
    Create comprehensive uncertainty heatmaps
    """
    # Prepare data for spatial visualization
    predictions = maps_data['predictions']
    epistemic = maps_data['epistemic_uncertainty']
    aleatoric = maps_data['aleatoric_uncertainty']
    total_unc = maps_data['total_uncertainty']
    confidence = maps_data['confidence']
    evidence_strength = maps_data['evidence_strength']
    
    # Determine grid size for spatial arrangement
    n_samples = len(predictions)
    grid_size = int(np.sqrt(n_samples))
    if grid_size * grid_size < n_samples:
        grid_size += 1
    
    # Pad data to fit grid
    pad_size = grid_size * grid_size - n_samples
    if pad_size > 0:
        predictions = np.pad(predictions, (0, pad_size), mode='constant', constant_values=np.nan)
        epistemic = np.pad(epistemic, (0, pad_size), mode='constant', constant_values=np.nan)
        aleatoric = np.pad(aleatoric, (0, pad_size), mode='constant', constant_values=np.nan)
        total_unc = np.pad(total_unc, (0, pad_size), mode='constant', constant_values=np.nan)
        confidence = np.pad(confidence, (0, pad_size), mode='constant', constant_values=np.nan)
        evidence_strength = np.pad(evidence_strength, (0, pad_size), mode='constant', constant_values=np.nan)
    
    # Reshape to 2D grids
    pred_grid = predictions.reshape(grid_size, grid_size)
    epistemic_grid = epistemic.reshape(grid_size, grid_size)
    aleatoric_grid = aleatoric.reshape(grid_size, grid_size)
    total_unc_grid = total_unc.reshape(grid_size, grid_size)
    confidence_grid = confidence.reshape(grid_size, grid_size)
    evidence_grid = evidence_strength.reshape(grid_size, grid_size)
    
    # Create comprehensive uncertainty visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'Evidential Uncertainty Maps - {dataset_name}', fontsize=16)
    
    # Custom colormap for uncertainty (white to red)
    uncertainty_cmap = LinearSegmentedColormap.from_list('uncertainty', ['white', 'yellow', 'red'])
    confidence_cmap = LinearSegmentedColormap.from_list('confidence', ['red', 'yellow', 'green'])
    
    # Predictions map
    im1 = axes[0, 0].imshow(pred_grid, cmap='RdYlBu_r', interpolation='nearest')
    axes[0, 0].set_title('Predictions (0=No Defect, 1=Defect)')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Epistemic uncertainty map
    im2 = axes[0, 1].imshow(epistemic_grid, cmap=uncertainty_cmap, interpolation='nearest')
    axes[0, 1].set_title('Epistemic Uncertainty (Model Uncertainty)')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Aleatoric uncertainty map
    im3 = axes[0, 2].imshow(aleatoric_grid, cmap=uncertainty_cmap, interpolation='nearest')
    axes[0, 2].set_title('Aleatoric Uncertainty (Data Uncertainty)')
    plt.colorbar(im3, ax=axes[0, 2])
    
    # Total uncertainty map
    im4 = axes[1, 0].imshow(total_unc_grid, cmap=uncertainty_cmap, interpolation='nearest')
    axes[1, 0].set_title('Total Uncertainty')
    plt.colorbar(im4, ax=axes[1, 0])
    
    # Confidence map
    im5 = axes[1, 1].imshow(confidence_grid, cmap=confidence_cmap, interpolation='nearest')
    axes[1, 1].set_title('Prediction Confidence')
    plt.colorbar(im5, ax=axes[1, 1])
    
    # Evidence strength map
    im6 = axes[1, 2].imshow(evidence_grid, cmap='viridis', interpolation='nearest')
    axes[1, 2].set_title('Evidence Strength (Alpha Sum)')
    plt.colorbar(im6, ax=axes[1, 2])
    
    # Remove axis ticks
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    filename = f'weights/evidential_uncertainty_maps_{dataset_name.replace(" ", "_").lower()}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Uncertainty maps saved to {filename}")


def create_defect_detection_maps(maps_data, dataset_name):
    """
    Create defect detection maps with uncertainty overlay
    """
    predictions = maps_data['predictions']
    total_unc = maps_data['total_uncertainty']
    confidence = maps_data['confidence']
    targets = maps_data['targets']
    
    # Determine grid size
    n_samples = len(predictions)
    grid_size = int(np.sqrt(n_samples))
    if grid_size * grid_size < n_samples:
        grid_size += 1
    
    # Pad data
    pad_size = grid_size * grid_size - n_samples
    if pad_size > 0:
        predictions = np.pad(predictions, (0, pad_size), mode='constant', constant_values=np.nan)
        total_unc = np.pad(total_unc, (0, pad_size), mode='constant', constant_values=np.nan)
        confidence = np.pad(confidence, (0, pad_size), mode='constant', constant_values=np.nan)
        if targets is not None:
            targets = np.pad(targets, (0, pad_size), mode='constant', constant_values=np.nan)
    
    # Reshape to grids
    pred_grid = predictions.reshape(grid_size, grid_size)
    unc_grid = total_unc.reshape(grid_size, grid_size)
    conf_grid = confidence.reshape(grid_size, grid_size)
    
    # Create visualization
    if targets is not None:
        target_grid = targets.reshape(grid_size, grid_size)
        fig, axes = plt.subplots(2, 2, figsize=(12, 12))
        fig.suptitle(f'Defect Detection with Uncertainty - {dataset_name}', fontsize=16)
        
        # Ground truth
        im1 = axes[0, 0].imshow(target_grid, cmap='RdYlBu_r', interpolation='nearest')
        axes[0, 0].set_title('Ground Truth')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Predictions
        im2 = axes[0, 1].imshow(pred_grid, cmap='RdYlBu_r', interpolation='nearest')
        axes[0, 1].set_title('Predictions')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # Uncertainty overlay on predictions
        axes[1, 0].imshow(pred_grid, cmap='RdYlBu_r', alpha=0.7, interpolation='nearest')
        im3 = axes[1, 0].imshow(unc_grid, cmap='Reds', alpha=0.5, interpolation='nearest')
        axes[1, 0].set_title('Predictions + Uncertainty Overlay')
        plt.colorbar(im3, ax=axes[1, 0])
        
        # High uncertainty regions
        high_unc_mask = unc_grid > (np.nanmean(unc_grid) + 2 * np.nanstd(unc_grid))
        axes[1, 1].imshow(pred_grid, cmap='RdYlBu_r', alpha=0.7, interpolation='nearest')
        axes[1, 1].imshow(high_unc_mask, cmap='Reds', alpha=0.8, interpolation='nearest')
        axes[1, 1].set_title('High Uncertainty Regions (>μ+2σ)')
        
    else:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'Defect Detection with Uncertainty - {dataset_name}', fontsize=16)
        
        # Predictions
        im1 = axes[0].imshow(pred_grid, cmap='RdYlBu_r', interpolation='nearest')
        axes[0].set_title('Predictions')
        plt.colorbar(im1, ax=axes[0])
        
        # Uncertainty overlay
        axes[1].imshow(pred_grid, cmap='RdYlBu_r', alpha=0.7, interpolation='nearest')
        im2 = axes[1].imshow(unc_grid, cmap='Reds', alpha=0.5, interpolation='nearest')
        axes[1].set_title('Predictions + Uncertainty Overlay')
        plt.colorbar(im2, ax=axes[1])
        
        # High uncertainty regions
        high_unc_mask = unc_grid > (np.nanmean(unc_grid) + 2 * np.nanstd(unc_grid))
        axes[2].imshow(pred_grid, cmap='RdYlBu_r', alpha=0.7, interpolation='nearest')
        axes[2].imshow(high_unc_mask, cmap='Reds', alpha=0.8, interpolation='nearest')
        axes[2].set_title('High Uncertainty Regions (>μ+2σ)')
    
    # Remove axis ticks
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    
    plt.tight_layout()
    filename = f'weights/evidential_defect_maps_{dataset_name.replace(" ", "_").lower()}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Defect detection maps saved to {filename}")


def create_uncertainty_method_comparison(methods_results):
    """
    Create visual comparison between different uncertainty methods
    """
    if len(methods_results) < 2:
        print("Need at least 2 methods for comparison")
        return
    
    fig, axes = plt.subplots(2, len(methods_results), figsize=(6*len(methods_results), 12))
    if len(methods_results) == 2:
        axes = axes.reshape(2, 2)
    
    fig.suptitle('Uncertainty Methods Comparison', fontsize=16)
    
    for i, (method, results) in enumerate(methods_results.items()):
        # Extract uncertainty data based on method
        if method == 'Evidential':
            uncertainty = results.get('total_uncertainty', results.get('uncertainties', None))
            predictions = results.get('predictions', None)
        else:
            uncertainty = results.get('uncertainties', results.get('total_uncertainty', None))
            predictions = results.get('predictions', None)
        
        if uncertainty is not None and predictions is not None:
            # Convert to numpy if needed
            if torch.is_tensor(uncertainty):
                uncertainty = uncertainty.cpu().numpy().flatten()
            if torch.is_tensor(predictions):
                predictions = predictions.cpu().numpy()
                if len(predictions.shape) > 1:
                    predictions = np.argmax(predictions, axis=-1).flatten()
            
            # Create spatial grid
            n_samples = len(uncertainty)
            grid_size = int(np.sqrt(n_samples))
            if grid_size * grid_size < n_samples:
                grid_size += 1
            
            pad_size = grid_size * grid_size - n_samples
            if pad_size > 0:
                uncertainty = np.pad(uncertainty, (0, pad_size), mode='constant', constant_values=np.nan)
                predictions = np.pad(predictions, (0, pad_size), mode='constant', constant_values=np.nan)
            
            unc_grid = uncertainty.reshape(grid_size, grid_size)
            pred_grid = predictions.reshape(grid_size, grid_size)
            
            # Uncertainty map
            im1 = axes[0, i].imshow(unc_grid, cmap='Reds', interpolation='nearest')
            axes[0, i].set_title(f'{method} - Uncertainty')
            axes[0, i].set_xticks([])
            axes[0, i].set_yticks([])
            plt.colorbar(im1, ax=axes[0, i])
            
            # Predictions with uncertainty overlay
            axes[1, i].imshow(pred_grid, cmap='RdYlBu_r', alpha=0.7, interpolation='nearest')
            im2 = axes[1, i].imshow(unc_grid, cmap='Reds', alpha=0.5, interpolation='nearest')
            axes[1, i].set_title(f'{method} - Predictions + Uncertainty')
            axes[1, i].set_xticks([])
            axes[1, i].set_yticks([])
            plt.colorbar(im2, ax=axes[1, i])
        else:
            axes[0, i].text(0.5, 0.5, f'{method}\nData not found', 
                           ha='center', va='center', transform=axes[0, i].transAxes)
            axes[1, i].text(0.5, 0.5, f'{method}\nData not found', 
                           ha='center', va='center', transform=axes[1, i].transAxes)
    
    plt.tight_layout()
    plt.savefig('weights/uncertainty_methods_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Uncertainty methods comparison saved to weights/uncertainty_methods_comparison.png")


def compare_uncertainty_methods():
    """
    Compare uncertainty estimates from different methods (if results exist)
    """
    methods = {
        'Evidential': 'weights/evidential_model_results.pth',
        'Bayesian': 'weights/bayesian_mcmc_model_results.pth', 
        'Dropout': 'uncertainty_simple_results/uncertainty_model_test_results.pth'
    }
    
    results = {}
    
    for method, path in methods.items():
        try:
            result = torch.load(path, map_location='cpu')
            results[method] = result
            print(f"Loaded {method} results")
        except FileNotFoundError:
            print(f"{method} results not found at {path}")
    
    if len(results) > 1:
        print("\nComparing uncertainty methods:")
        for method, result in results.items():
            if 'test_accuracy' in result:
                print(f"{method} - Test Accuracy: {result['test_accuracy']:.2f}%")
            elif 'accuracy' in result:
                accuracy = result['accuracy']
                if torch.is_tensor(accuracy):
                    accuracy = accuracy.item()
                print(f"{method} - Test Accuracy: {accuracy*100:.2f}%")
        
        # Create visual comparison
        create_uncertainty_method_comparison(results)
    
    return results


if __name__ == '__main__':
    print("=== Evidential Deep Learning Analysis and Mapping ===\n")
    
    # Test if evidential results exist
    results_path = 'weights/evidential_model_results.pth'
    model_path = 'weights/evidential_model.pth'
    
    try:
        # First, try to analyze saved results
        print("1. Analyzing evidential results...")
        results = analyze_evidential_results(results_path)
        
        # Generate comprehensive uncertainty maps
        print("\n2. Generating uncertainty maps...")
        all_maps = generate_evidential_uncertainty_maps(model_path, save_maps=True)
        
        # Compare with other uncertainty methods
        print("\n3. Comparing uncertainty methods...")
        all_results = compare_uncertainty_methods()
        
        print("\n=== Analysis Complete ===")
        print("Generated maps:")
        for dataset_name in all_maps.keys():
            safe_name = dataset_name.replace(" ", "_").lower()
            print(f"  - Uncertainty maps: weights/evidential_uncertainty_maps_{safe_name}.png")
            print(f"  - Defect detection maps: weights/evidential_defect_maps_{safe_name}.png")
        
    except FileNotFoundError as e:
        print(f"Results file not found: {e}")
        print("Attempting to generate maps with trained model...")
        
        try:
            # Generate maps even without results file
            print("\nGenerating uncertainty maps from trained model...")
            all_maps = generate_evidential_uncertainty_maps(model_path, save_maps=True)
            
            print("\n=== Map Generation Complete ===")
            print("Generated maps:")
            for dataset_name in all_maps.keys():
                safe_name = dataset_name.replace(" ", "_").lower()
                print(f"  - Uncertainty maps: weights/evidential_uncertainty_maps_{safe_name}.png")
                print(f"  - Defect detection maps: weights/evidential_defect_maps_{safe_name}.png")
                
        except FileNotFoundError:
            print(f"Model file not found at {model_path}")
            print("Please train the evidential model first using train_evidential.py")
            
            # Show what would be generated
            print("\nThe following maps would be generated after training:")
            print("  - Epistemic uncertainty maps (model uncertainty)")
            print("  - Aleatoric uncertainty maps (data uncertainty)") 
            print("  - Total uncertainty maps")
            print("  - Confidence maps")
            print("  - Evidence strength maps")
            print("  - Defect detection maps with uncertainty overlay")
            print("  - High uncertainty region identification")