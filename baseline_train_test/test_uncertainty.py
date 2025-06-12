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

# configure logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


class UncertaintyIENet(nn.Module):
    """
    IENet with uncertainty estimation capabilities.
    Implements both aleatoric (data) and epistemic (model) uncertainty.
    """
    
    def __init__(self, dropout_rate=0.2, verbose=False):
        super(UncertaintyIENet, self).__init__()
        self.verbose = verbose
        self.dropout_rate = dropout_rate
        
        # Feature extraction layers (same as baseline)
        self.residual_1 = ResidualBlock(1, 8, 200)
        self.residual_2 = ResidualBlock(8, 16, 100)
        self.residual_3 = ResidualBlock(16, 16, 50)
        self.residual_4 = ResidualBlock(16, 32, 25)
        self.residual_5 = ResidualBlock(32, 64, 13)
        self.residual_6 = ResidualBlock(64, 64, 7)
        
        # LSTM layers with dropout for epistemic uncertainty
        self.bilstm_1 = nn.LSTM(input_size=832, hidden_size=32, num_layers=1, 
                               batch_first=True, bidirectional=True, dropout=dropout_rate)
        self.bilstm_2 = nn.LSTM(input_size=64, hidden_size=32, num_layers=1, 
                               batch_first=True, bidirectional=True, dropout=dropout_rate)
        self.bilstm_3 = nn.LSTM(input_size=64, hidden_size=32, num_layers=1, 
                               batch_first=True, bidirectional=True, dropout=dropout_rate)
        
        # Dropout layers for epistemic uncertainty
        self.dropout1 = nn.Dropout(dropout_rate)
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Classification head - outputs mean logits
        self.classifier_mean = nn.Linear(in_features=64, out_features=2)
        
        # Aleatoric uncertainty head - outputs log variance
        self.classifier_logvar = nn.Linear(in_features=64, out_features=2)

    def forward(self, x):
        # Feature extraction
        x = self.residual_1(x)
        x = self.residual_2(x)
        x = self.residual_3(x)
        x = self.residual_4(x)
        x = self.residual_5(x)
        x = self.residual_6(x)

        # Reshape and flatten
        x = x.view(x.size(0), -1)
        x = nn.Flatten()(x)
        x = x.unsqueeze(0)

        # LSTM layers with dropout
        x, _ = self.bilstm_1(x)
        x = self.dropout1(x)
        x, _ = self.bilstm_2(x)
        x = self.dropout2(x)
        features, _ = self.bilstm_3(x)
        
        # Output predictions
        mean_logits = self.classifier_mean(features)
        log_var = self.classifier_logvar(features)
        
        return mean_logits, log_var, features

    def predict_with_uncertainty(self, x, mc_samples=50):
        """
        Perform Monte Carlo sampling to estimate epistemic uncertainty
        """
        self.train()  # Enable dropout
        
        predictions = []
        aleatoric_vars = []
        
        with torch.no_grad():
            for _ in range(mc_samples):
                mean_logits, log_var, _ = self.forward(x)
                predictions.append(F.softmax(mean_logits, dim=-1))
                aleatoric_vars.append(torch.exp(log_var))
        
        # Stack predictions
        predictions = torch.stack(predictions)  # [mc_samples, batch_size, seq_len, num_classes]
        aleatoric_vars = torch.stack(aleatoric_vars)
        
        # Calculate epistemic uncertainty (variance across MC samples)
        mean_prediction = torch.mean(predictions, dim=0)
        epistemic_uncertainty = torch.var(predictions, dim=0)
        
        # Calculate aleatoric uncertainty (average of predicted variances)
        aleatoric_uncertainty = torch.mean(aleatoric_vars, dim=0)
        
        # Total uncertainty
        total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
        
        return mean_prediction, epistemic_uncertainty, aleatoric_uncertainty, total_uncertainty


def create_uncertainty_maps(predictions, epistemic_unc, aleatoric_unc, total_unc, 
                          shape, dataset_name, model_name, save_individual=True):
    """
    Create and save uncertainty visualization maps
    """
    # Extract class 0 probabilities (non-defect probability)
    pred_map = predictions.squeeze(0)[:, 0].cpu().detach().numpy()
    
    # Extract uncertainties for class 0
    epistemic_map = epistemic_unc.squeeze(0)[:, 0].cpu().detach().numpy()
    aleatoric_map = aleatoric_unc.squeeze(0)[:, 0].cpu().detach().numpy()
    total_unc_map = total_unc.squeeze(0)[:, 0].cpu().detach().numpy()
    
    # Reshape to spatial dimensions
    pred_reshaped = np.reshape(pred_map, shape)
    epistemic_reshaped = np.reshape(epistemic_map, shape)
    aleatoric_reshaped = np.reshape(aleatoric_map, shape)
    total_unc_reshaped = np.reshape(total_unc_map, shape)
    
    if save_individual:
        # Classification map
        plt.figure(figsize=(8, 6))
        plt.imshow(pred_reshaped, cmap='Spectral', interpolation='hamming')
        plt.colorbar(label='Non-defect Probability')
        plt.title(f'{dataset_name} - Classification')
        plt.axis('off')
        plt.savefig(f'{model_name}_{dataset_name}_classification.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Epistemic uncertainty map
        plt.figure(figsize=(8, 6))
        plt.imshow(epistemic_reshaped, cmap='Reds', interpolation='hamming')
        plt.colorbar(label='Epistemic Uncertainty')
        plt.title(f'{dataset_name} - Epistemic Uncertainty (Model)')
        plt.axis('off')
        plt.savefig(f'{model_name}_{dataset_name}_epistemic.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Aleatoric uncertainty map
        plt.figure(figsize=(8, 6))
        plt.imshow(aleatoric_reshaped, cmap='Blues', interpolation='hamming')
        plt.colorbar(label='Aleatoric Uncertainty')
        plt.title(f'{dataset_name} - Aleatoric Uncertainty (Data)')
        plt.axis('off')
        plt.savefig(f'{model_name}_{dataset_name}_aleatoric.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Total uncertainty map
        plt.figure(figsize=(8, 6))
        plt.imshow(total_unc_reshaped, cmap='Purples', interpolation='hamming')
        plt.colorbar(label='Total Uncertainty')
        plt.title(f'{dataset_name} - Total Uncertainty')
        plt.axis('off')
        plt.savefig(f'{model_name}_{dataset_name}_total_uncertainty.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # Combined visualization
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{dataset_name} - Classification and Uncertainty Analysis', fontsize=16)
    
    # Classification
    im1 = axes[0, 0].imshow(pred_reshaped, cmap='Spectral', interpolation='hamming')
    axes[0, 0].set_title('Classification (Non-defect Prob.)')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0])
    
    # Epistemic uncertainty
    im2 = axes[0, 1].imshow(epistemic_reshaped, cmap='Reds', interpolation='hamming')
    axes[0, 1].set_title('Epistemic Uncertainty (Model)')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1])
    
    # Aleatoric uncertainty
    im3 = axes[1, 0].imshow(aleatoric_reshaped, cmap='Blues', interpolation='hamming')
    axes[1, 0].set_title('Aleatoric Uncertainty (Data)')
    axes[1, 0].axis('off')
    plt.colorbar(im3, ax=axes[1, 0])
    
    # Total uncertainty
    im4 = axes[1, 1].imshow(total_unc_reshaped, cmap='Purples', interpolation='hamming')
    axes[1, 1].set_title('Total Uncertainty')
    axes[1, 1].axis('off')
    plt.colorbar(im4, ax=axes[1, 1])
    
    plt.tight_layout()
    plt.savefig(f'{model_name}_{dataset_name}_combined.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return pred_reshaped, epistemic_reshaped, aleatoric_reshaped, total_unc_reshaped


def analyze_uncertainty_statistics(predictions, epistemic_unc, aleatoric_unc, total_unc, dataset_name):
    """
    Analyze and print uncertainty statistics
    """
    pred_classes = torch.argmax(predictions.squeeze(0), dim=-1)
    confidence = torch.max(F.softmax(predictions.squeeze(0), dim=-1), dim=-1)[0]
    
    print(f"\n=== {dataset_name} Statistics ===")
    print(f"Samples: {len(pred_classes)}")
    print(f"Predicted Defects: {(pred_classes == 1).sum().item()}")
    print(f"Predicted Non-defects: {(pred_classes == 0).sum().item()}")
    print(f"Mean Confidence: {confidence.mean():.4f} ± {confidence.std():.4f}")
    
    epistemic_mean = epistemic_unc.mean()
    aleatoric_mean = aleatoric_unc.mean()
    total_mean = total_unc.mean()
    
    print(f"Mean Epistemic Uncertainty: {epistemic_mean:.4f} ± {epistemic_unc.std():.4f}")
    print(f"Mean Aleatoric Uncertainty: {aleatoric_mean:.4f} ± {aleatoric_unc.std():.4f}")
    print(f"Mean Total Uncertainty: {total_mean:.4f} ± {total_unc.std():.4f}")
    
    # High uncertainty samples
    high_unc_threshold = total_mean + 2 * total_unc.std()
    high_unc_samples = (total_unc.squeeze(0)[:, 0] > high_unc_threshold).sum()
    print(f"High Uncertainty Samples (>μ+2σ): {high_unc_samples}")


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
    
    model_name = 'uncertainty_model_class_weight'
    mc_samples = 50  # Number of MC samples for uncertainty estimation
    
    logger.info(f"Using device: {device}")
    logger.info(f"Using uncertainty model: {model_name}")
    logger.info(f"MC samples for uncertainty: {mc_samples}")
    
    # Load uncertainty model
    classifier = UncertaintyIENet(dropout_rate=0.3, verbose=False).to(device)
    classifier.load_state_dict(torch.load(f'weights/{model_name}.pth', map_location=device))
    
    # Test on DS1 (with ground truth)
    logger.info("Testing on DS1...")
    pred1, epistemic1, aleatoric1, total1 = classifier.predict_with_uncertainty(X_ds1, mc_samples)
    
    # Calculate accuracy for DS1
    pred_classes_ds1 = torch.argmax(pred1.squeeze(0), dim=-1)
    accuracy_ds1 = (pred_classes_ds1.cpu() == torch.tensor(y_ds1)).float().mean()
    logger.info(f"DS1 Accuracy: {accuracy_ds1:.4f}")
    
    # Create DS1 maps
    pred_map1, epi_map1, ale_map1, tot_map1 = create_uncertainty_maps(
        pred1, epistemic1, aleatoric1, total1, (9, 28), 'ds1', model_name
    )
    analyze_uncertainty_statistics(pred1, epistemic1, aleatoric1, total1, 'DS1')
    logger.info("DS1 - Test Maps Generated")
    
    # Test on CCNY May data
    logger.info("Testing on CCNY May 2022...")
    pred2, epistemic2, aleatoric2, total2 = classifier.predict_with_uncertainty(X_may, mc_samples)
    
    pred_map2, epi_map2, ale_map2, tot_map2 = create_uncertainty_maps(
        pred2, epistemic2, aleatoric2, total2, (31, 38), 'ccny_may', model_name
    )
    analyze_uncertainty_statistics(pred2, epistemic2, aleatoric2, total2, 'CCNY May 2022')
    logger.info("CCNY May 2022 - Test Maps Generated")
    
    # Test on CCNY June data
    logger.info("Testing on CCNY June 2022...")
    pred3, epistemic3, aleatoric3, total3 = classifier.predict_with_uncertainty(X_june, mc_samples)
    
    pred_map3, epi_map3, ale_map3, tot_map3 = create_uncertainty_maps(
        pred3, epistemic3, aleatoric3, total3, (19, 34), 'ccny_june', model_name
    )
    analyze_uncertainty_statistics(pred3, epistemic3, aleatoric3, total3, 'CCNY June 2022')
    logger.info("CCNY June 2022 - Test Maps Generated")
    
    # Test on CCNY Nov 2023 data
    logger.info("Testing on CCNY Nov 2023...")
    pred4, epistemic4, aleatoric4, total4 = classifier.predict_with_uncertainty(X_nov23, mc_samples)
    
    pred_map4, epi_map4, ale_map4, tot_map4 = create_uncertainty_maps(
        pred4, epistemic4, aleatoric4, total4, (44, 34), 'ccny_nov2023', model_name
    )
    analyze_uncertainty_statistics(pred4, epistemic4, aleatoric4, total4, 'CCNY Nov 2023')
    logger.info("CCNY Nov 2023 - Test Maps Generated")
    
    # Save all results
    results = {
        'ds1': {
            'predictions': pred1.cpu(),
            'epistemic_uncertainty': epistemic1.cpu(),
            'aleatoric_uncertainty': aleatoric1.cpu(),
            'total_uncertainty': total1.cpu(),
            'ground_truth': y_ds1,
            'accuracy': accuracy_ds1,
        },
        'ccny_may': {
            'predictions': pred2.cpu(),
            'epistemic_uncertainty': epistemic2.cpu(),
            'aleatoric_uncertainty': aleatoric2.cpu(),
            'total_uncertainty': total2.cpu(),
        },
        'ccny_june': {
            'predictions': pred3.cpu(),
            'epistemic_uncertainty': epistemic3.cpu(),
            'aleatoric_uncertainty': aleatoric3.cpu(),
            'total_uncertainty': total3.cpu(),
        },
        'ccny_nov2023': {
            'predictions': pred4.cpu(),
            'epistemic_uncertainty': epistemic4.cpu(),
            'aleatoric_uncertainty': aleatoric4.cpu(),
            'total_uncertainty': total4.cpu(),
        }
    }
    
    torch.save(results, f'weights/{model_name}_test_results.pth')
    logger.info(f"All results saved to weights/{model_name}_test_results.pth")
    
    print("\n=== Summary ===")
    print(f"DS1 Accuracy: {accuracy_ds1:.4f}")
    print("Generated maps for all datasets with classification and uncertainty visualizations")
    print("Individual and combined visualization maps saved as PNG files")