import warnings
warnings.filterwarnings("ignore")

import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, '/Users/evhoxha/projects/impact_echo_contrastive_learning/')
sys.path.insert(0, 'dataloaders/')
sys.path.insert(0, 'models/')
sys.path.insert(0, 'data/')

from utils import *
from models.model_parts import ResidualBlock
from train_bayesian_mcmc import BayesianLinear, BayesianIENet

# configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def create_bayesian_uncertainty_maps(predictions, uncertainties, confidences, 
                                   shape, dataset_name, model_name, save_individual=True):
    """
    Create and save Bayesian uncertainty visualization maps
    """
    # Extract class 0 probabilities (non-defect probability)
    pred_map = predictions.squeeze(0)[:, 0].cpu().detach().numpy()
    
    # Extract uncertainties and confidences for class 0
    uncertainty_map = uncertainties.squeeze(0)[:, 0].cpu().detach().numpy()
    confidence_map = confidences.squeeze(0).cpu().detach().numpy()
    
    # Reshape to spatial dimensions
    pred_reshaped = np.reshape(pred_map, shape)
    uncertainty_reshaped = np.reshape(uncertainty_map, shape)
    confidence_reshaped = np.reshape(confidence_map, shape)
    
    if save_individual:
        # Classification map
        plt.figure(figsize=(8, 6))
        plt.imshow(pred_reshaped, cmap='Spectral', interpolation='hamming')
        plt.colorbar(label='Non-defect Probability')
        plt.title(f'{dataset_name} - Bayesian Classification')
        plt.axis('off')
        plt.savefig(f'{model_name}_{dataset_name}_bayesian_classification.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Predictive uncertainty map
        plt.figure(figsize=(8, 6))
        plt.imshow(uncertainty_reshaped, cmap='Reds', interpolation='hamming')
        plt.colorbar(label='Predictive Uncertainty')
        plt.title(f'{dataset_name} - Bayesian Predictive Uncertainty')
        plt.axis('off')
        plt.savefig(f'{model_name}_{dataset_name}_predictive_uncertainty.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Confidence map
        plt.figure(figsize=(8, 6))
        plt.imshow(confidence_reshaped, cmap='Blues', interpolation='hamming')
        plt.colorbar(label='Prediction Confidence')
        plt.title(f'{dataset_name} - Bayesian Confidence')
        plt.axis('off')
        plt.savefig(f'{model_name}_{dataset_name}_confidence.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # Combined visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'{dataset_name} - Bayesian Analysis', fontsize=16)
    
    # Classification
    im1 = axes[0].imshow(pred_reshaped, cmap='Spectral', interpolation='hamming')
    axes[0].set_title('Classification (Non-defect Prob.)')
    axes[0].axis('off')
    plt.colorbar(im1, ax=axes[0])
    
    # Predictive uncertainty
    im2 = axes[1].imshow(uncertainty_reshaped, cmap='Reds', interpolation='hamming')
    axes[1].set_title('Predictive Uncertainty')
    axes[1].axis('off')
    plt.colorbar(im2, ax=axes[1])
    
    # Confidence
    im3 = axes[2].imshow(confidence_reshaped, cmap='Blues', interpolation='hamming')
    axes[2].set_title('Prediction Confidence')
    axes[2].axis('off')
    plt.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_{dataset_name}_bayesian_combined.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return pred_reshaped, uncertainty_reshaped, confidence_reshaped


def analyze_bayesian_statistics(predictions, uncertainties, confidences, dataset_name):
    """
    Analyze and print Bayesian uncertainty statistics
    """
    pred_classes = torch.argmax(predictions.squeeze(0), dim=-1)
    
    print(f"\n=== {dataset_name} Bayesian Statistics ===")
    print(f"Samples: {len(pred_classes)}")
    print(f"Predicted Defects: {(pred_classes == 1).sum().item()}")
    print(f"Predicted Non-defects: {(pred_classes == 0).sum().item()}")
    
    mean_confidence = confidences.mean()
    mean_uncertainty = uncertainties.mean()
    
    print(f"Mean Confidence: {mean_confidence:.4f} ± {confidences.std():.4f}")
    print(f"Mean Predictive Uncertainty: {mean_uncertainty:.4f} ± {uncertainties.std():.4f}")
    
    # High uncertainty samples
    high_unc_threshold = mean_uncertainty + 2 * uncertainties.std()
    high_unc_samples = (uncertainties.squeeze(0)[:, 0] > high_unc_threshold).sum()
    print(f"High Uncertainty Samples (>μ+2σ): {high_unc_samples}")
    
    # Low confidence samples
    low_conf_threshold = mean_confidence - 2 * confidences.std()
    low_conf_samples = (confidences.squeeze(0) < low_conf_threshold).sum()
    print(f"Low Confidence Samples (<μ-2σ): {low_conf_samples}")


def compare_uncertainty_methods(bayesian_results, uncertainty_results, dataset_name):
    """
    Compare Bayesian vs Dropout-based uncertainty estimation
    """
    print(f"\n=== {dataset_name} Method Comparison ===")
    
    # Bayesian results
    bayesian_pred = bayesian_results['predictions']
    bayesian_unc = bayesian_results['uncertainties']
    bayesian_conf = bayesian_results['confidences']
    
    # Dropout results (if available)
    if uncertainty_results is not None:
        dropout_pred = uncertainty_results['predictions']
        dropout_total_unc = uncertainty_results['total_uncertainty']
        
        # Compare predictions
        bayesian_classes = torch.argmax(bayesian_pred.squeeze(0), dim=-1)
        dropout_classes = torch.argmax(dropout_pred.squeeze(0), dim=-1)
        
        agreement = (bayesian_classes == dropout_classes).float().mean()
        print(f"Prediction Agreement: {agreement:.4f}")
        
        # Compare uncertainties (correlation)
        bayesian_unc_flat = bayesian_unc.squeeze(0)[:, 0].flatten()
        dropout_unc_flat = dropout_total_unc.squeeze(0)[:, 0].flatten()
        
        correlation = torch.corrcoef(torch.stack([bayesian_unc_flat, dropout_unc_flat]))[0, 1]
        print(f"Uncertainty Correlation: {correlation:.4f}")
        
        print(f"Bayesian Mean Uncertainty: {bayesian_unc.mean():.4f}")
        print(f"Dropout Mean Uncertainty: {dropout_total_unc.mean():.4f}")
    
    print(f"Bayesian Mean Confidence: {bayesian_conf.mean():.4f}")


if __name__ == '__main__':
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Load test datasets
    X_ds1, y_ds1 = load_ds1_test_data_into_torch_tensor(
        device=device, 
        X_path='data/X_test_860.npy', 
        y_path='data/y_test.npy'
    )
    X_may, X_june = load_ccny_sep2022_data_into_torch_tensor(
        device=device,
        X_path='data/X_our_slab_size860.npy'
    )
    X_nov23 = load_ccny_nov2023_data_into_torch_tensor2(
        device=device,
        X_path='data/nov2023_non_resampled.npy'
    )
    
    model_name = 'bayesian_mcmc_model'
    n_samples = 100  # Number of samples for Bayesian inference
    
    logger.info(f"Using device: {device}")
    logger.info(f"Using Bayesian model: {model_name}")
    logger.info(f"Bayesian samples for inference: {n_samples}")
    
    # Load Bayesian model
    bayesian_model = BayesianIENet(prior_var=1.0, verbose=False).to(device)
    bayesian_model.load_state_dict(torch.load(f'weights/{model_name}.pth', map_location=device))
    
    # Try to load dropout-based uncertainty results for comparison
    uncertainty_results = None
    try:
        uncertainty_results = torch.load('weights/uncertainty_model_test_results.pth', map_location=device)
        logger.info("Loaded dropout-based uncertainty results for comparison")
    except FileNotFoundError:
        logger.info("No dropout-based uncertainty results found for comparison")
    
    # Test on DS1 (with ground truth)
    logger.info("Testing Bayesian model on DS1...")
    pred1, unc1, conf1 = bayesian_model.predict_with_uncertainty(X_ds1, n_samples)
    
    # Calculate accuracy for DS1
    pred_classes_ds1 = torch.argmax(pred1.squeeze(0), dim=-1)
    accuracy_ds1 = (pred_classes_ds1.cpu() == torch.tensor(y_ds1)).float().mean()
    logger.info(f"DS1 Bayesian Accuracy: {accuracy_ds1:.4f}")
    
    # Create DS1 maps
    pred_map1, unc_map1, conf_map1 = create_bayesian_uncertainty_maps(
        pred1, unc1, conf1, (9, 28), 'ds1', model_name
    )
    analyze_bayesian_statistics(pred1, unc1, conf1, 'DS1')
    
    # Compare with dropout method if available
    if uncertainty_results and 'ds1' in uncertainty_results:
        compare_uncertainty_methods(
            {'predictions': pred1, 'uncertainties': unc1, 'confidences': conf1},
            uncertainty_results['ds1'], 'DS1'
        )
    
    logger.info("DS1 - Bayesian Test Maps Generated")
    
    # Test on CCNY May data
    logger.info("Testing Bayesian model on CCNY May 2022...")
    pred2, unc2, conf2 = bayesian_model.predict_with_uncertainty(X_may, n_samples)
    
    pred_map2, unc_map2, conf_map2 = create_bayesian_uncertainty_maps(
        pred2, unc2, conf2, (31, 38), 'ccny_may', model_name
    )
    analyze_bayesian_statistics(pred2, unc2, conf2, 'CCNY May 2022')
    
    if uncertainty_results and 'ccny_may' in uncertainty_results:
        compare_uncertainty_methods(
            {'predictions': pred2, 'uncertainties': unc2, 'confidences': conf2},
            uncertainty_results['ccny_may'], 'CCNY May 2022'
        )
    
    logger.info("CCNY May 2022 - Bayesian Test Maps Generated")
    
    # Test on CCNY June data
    logger.info("Testing Bayesian model on CCNY June 2022...")
    pred3, unc3, conf3 = bayesian_model.predict_with_uncertainty(X_june, n_samples)
    
    pred_map3, unc_map3, conf_map3 = create_bayesian_uncertainty_maps(
        pred3, unc3, conf3, (19, 34), 'ccny_june', model_name
    )
    analyze_bayesian_statistics(pred3, unc3, conf3, 'CCNY June 2022')
    
    if uncertainty_results and 'ccny_june' in uncertainty_results:
        compare_uncertainty_methods(
            {'predictions': pred3, 'uncertainties': unc3, 'confidences': conf3},
            uncertainty_results['ccny_june'], 'CCNY June 2022'
        )
    
    logger.info("CCNY June 2022 - Bayesian Test Maps Generated")
    
    # Test on CCNY Nov 2023 data
    logger.info("Testing Bayesian model on CCNY Nov 2023...")
    pred4, unc4, conf4 = bayesian_model.predict_with_uncertainty(X_nov23, n_samples)
    
    pred_map4, unc_map4, conf_map4 = create_bayesian_uncertainty_maps(
        pred4, unc4, conf4, (44, 34), 'ccny_nov2023', model_name
    )
    analyze_bayesian_statistics(pred4, unc4, conf4, 'CCNY Nov 2023')
    
    if uncertainty_results and 'ccny_nov2023' in uncertainty_results:
        compare_uncertainty_methods(
            {'predictions': pred4, 'uncertainties': unc4, 'confidences': conf4},
            uncertainty_results['ccny_nov2023'], 'CCNY Nov 2023'
        )
    
    logger.info("CCNY Nov 2023 - Bayesian Test Maps Generated")
    
    # Save all Bayesian results
    bayesian_results = {
        'ds1': {
            'predictions': pred1.cpu(),
            'uncertainties': unc1.cpu(),
            'confidences': conf1.cpu(),
            'ground_truth': y_ds1,
            'accuracy': accuracy_ds1,
        },
        'ccny_may': {
            'predictions': pred2.cpu(),
            'uncertainties': unc2.cpu(),
            'confidences': conf2.cpu(),
        },
        'ccny_june': {
            'predictions': pred3.cpu(),
            'uncertainties': unc3.cpu(),
            'confidences': conf3.cpu(),
        },
        'ccny_nov2023': {
            'predictions': pred4.cpu(),
            'uncertainties': unc4.cpu(),
            'confidences': conf4.cpu(),
        }
    }
    
    torch.save(bayesian_results, f'weights/{model_name}_test_results.pth')
    logger.info(f"All Bayesian results saved to weights/{model_name}_test_results.pth")
    
    print("\n=== Bayesian Summary ===")
    print(f"DS1 Accuracy: {accuracy_ds1:.4f}")
    print("Generated Bayesian uncertainty maps for all datasets")
    print("Bayesian approach learns uncertainty through weight distributions")
    print("Individual and combined visualization maps saved as PNG files")
    
    # Summary statistics across all datasets
    all_uncertainties = torch.cat([
        unc1.squeeze(0)[:, 0].flatten(),
        unc2.squeeze(0)[:, 0].flatten(),
        unc3.squeeze(0)[:, 0].flatten(),
        unc4.squeeze(0)[:, 0].flatten()
    ])
    
    all_confidences = torch.cat([
        conf1.squeeze(0).flatten(),
        conf2.squeeze(0).flatten(),
        conf3.squeeze(0).flatten(),
        conf4.squeeze(0).flatten()
    ])
    
    print(f"\nOverall Statistics:")
    print(f"Global Mean Uncertainty: {all_uncertainties.mean():.4f} ± {all_uncertainties.std():.4f}")
    print(f"Global Mean Confidence: {all_confidences.mean():.4f} ± {all_confidences.std():.4f}")