import warnings
warnings.filterwarnings("ignore")
import sys
sys.path.insert(0, '/Users/evhoxha/projects/impact_echo_contrastive_learning/')
sys.path.insert(0, 'dataloaders/')
sys.path.insert(0, 'models/')
sys.path.insert(0, 'data/')

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from dataloaders.dataloader import ImpactEchoDatasetClassifier, ImpactEchoDatasetClassifierAug
import tqdm
import numpy as np
from utils import *

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

from models.model_parts import ResidualBlock
from torch_geometric.nn import MLP


import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class WaveFeatureExtractor(nn.Module):
    """Multi-scale wave feature extraction with dilated convolutions"""
    def __init__(self, in_channels, out_channels, kernel_sizes=[200, 100, 50, 25, 13, 7]):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels // len(kernel_sizes), 
                         kernel_size=k, padding='same', dilation=1),
                nn.BatchNorm1d(out_channels // len(kernel_sizes)),
                nn.GELU()
            ) for k in kernel_sizes
        ])
        
    def forward(self, x):
        # Multi-scale feature extraction
        features = [branch(x) for branch in self.branches]
        return torch.cat(features, dim=1)

class EnhancedResidualBlock(nn.Module):
    """Residual block with SE attention and better gradient flow"""
    def __init__(self, in_channels, out_channels, seq_len, reduction=4):
        super().__init__()
        self.downsample = nn.AvgPool1d(2) if seq_len > 1 else nn.Identity()
        mid_channels = out_channels
        
        self.conv1 = nn.Conv1d(in_channels, mid_channels, 3, padding=1)
        self.bn1 = nn.BatchNorm1d(mid_channels)
        self.conv2 = nn.Conv1d(mid_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # Squeeze-and-Excitation for channel attention
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Conv1d(out_channels, out_channels // reduction, 1),
            nn.ReLU(),
            nn.Conv1d(out_channels // reduction, out_channels, 1),
            nn.Sigmoid()
        )
        
        # Skip connection
        self.skip = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.activation = nn.GELU()
        
    def forward(self, x):
        identity = self.skip(x)
        
        out = self.activation(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Channel attention
        att = self.se(out)
        out = out * att
        
        out = out + identity
        out = self.activation(out)
        out = self.downsample(out)
        
        return out

class PositionalEncoding(nn.Module):
    """Learnable positional encoding for wave signals"""
    def __init__(self, d_model, max_len=256):
        super().__init__()
        self.pe = nn.Parameter(torch.randn(1, d_model, max_len) * 0.02)
        
    def forward(self, x):
        # x shape: (batch, channels, seq_len)
        return x + self.pe[:, :, :x.size(2)]

class ImprovedEvidentialIENet(nn.Module):
    """
    Enhanced Evidential IENet for 1D wave signal defect detection
    """
    def __init__(self, num_classes=2, input_length=200, dropout=0.1):
        super().__init__()
        self.num_classes = num_classes
        
        # Multi-scale initial feature extraction
        self.wave_features = WaveFeatureExtractor(1, 36)
        
        # Progressive feature refinement with better gradient flow
        self.residual_1 = EnhancedResidualBlock(36, 64, input_length)
        self.residual_2 = EnhancedResidualBlock(64, 128, input_length // 2)
        self.residual_3 = EnhancedResidualBlock(128, 256, input_length // 4)
        self.residual_4 = EnhancedResidualBlock(256, 256, input_length // 8)
        
        # Positional encoding for transformer
        self.pos_encoding = PositionalEncoding(256)
        
        # Efficient transformer with appropriate heads
        # 8 heads × 32 dims = 256 total dims
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256, 
            nhead=8,  # More reasonable: 256/8 = 32 dims per head
            dim_feedforward=1024,
            dropout=dropout,
            activation='gelu',
            batch_first=True  # More intuitive
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Global and local feature aggregation
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.local_pool = nn.AdaptiveMaxPool1d(4)
        
        # Calculate feature dimension: 256 (global) + 256*4 (local) = 1280
        feature_dim = 256 + 256 * 4
        
        # Evidence pathway with uncertainty decomposition
        self.evidence_pathway = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Separate evidence branches for aleatoric and epistemic uncertainty
        self.evidence_layer = nn.Linear(256, num_classes)
        self.uncertainty_layer = nn.Linear(256, num_classes)  # For epistemic uncertainty
        
        # Auxiliary defect characteristic predictor (size, depth, type)
        self.defect_features = nn.Linear(256, 16)  # Additional defect characteristics
        
    def forward(self, x):
        # x shape: (batch, 1, seq_len)
        
        # Multi-scale wave feature extraction
        x = self.wave_features(x)
        
        # Hierarchical feature extraction
        x = self.residual_1(x)
        x = self.residual_2(x)
        x = self.residual_3(x)
        x = self.residual_4(x)
        
        # Add positional encoding for transformer
        x = self.pos_encoding(x)
        
        # Transformer processing (batch, channels, seq_len) -> (batch, seq_len, channels)
        x_t = x.transpose(1, 2)
        x_t = self.transformer(x_t)
        x = x_t.transpose(1, 2)
        
        # Multi-level pooling
        global_features = self.global_pool(x).squeeze(-1)  # (batch, 256)
        local_features = self.local_pool(x).flatten(1)     # (batch, 256*4)
        
        # Combine global and local features
        combined = torch.cat([global_features, local_features], dim=1)
        
        # Evidence pathway
        features = self.evidence_pathway(combined)
        
        # Evidence outputs (Dirichlet parameters)
        evidence = F.softplus(self.evidence_layer(features)) + 1e-6
        uncertainty = F.softplus(self.uncertainty_layer(features)) + 1e-6
        
        # Defect characteristics (optional auxiliary output)
        defect_features = self.defect_features(features)
        
        return evidence, defect_features

    def predict_with_uncertainty(self, x):
        """
        Get predictions with evidential uncertainty measures
        """
        with torch.no_grad():
            evidence, _ = self.forward(x)
            
            # Dirichlet parameters (alpha = evidence + 1)
            alphas = evidence + 1.0
            
            # Sum of alphas (strength of Dirichlet)
            alpha_sum = torch.sum(alphas, dim=-1, keepdim=True)
            
            # Expected probabilities (mean of Dirichlet)
            prob = alphas / alpha_sum
            
            # Uncertainty measures
            aleatoric_uncertainty, epistemic_uncertainty, total_uncertainty = ImprovedEvidentialIENet.compute_uncertainties(alphas)
            
            # 4. Confidence (max probability)
            confidence = torch.max(prob, dim=-1)[0]
            
            return prob, epistemic_uncertainty, aleatoric_uncertainty, total_uncertainty, confidence, alpha_sum

    @staticmethod
    def compute_uncertainties(alpha):
        S = torch.sum(alpha, dim=-1, keepdim=True)
        mean = alpha / S
        aleatoric = -torch.sum(mean * (torch.digamma(alpha + 1) - torch.digamma(S + 1)), dim=-1)
        total = -torch.sum(mean * torch.log(mean + 1e-10), dim=-1)
        epistemic = total - aleatoric
        return aleatoric, epistemic, total


def dirichlet_kl_divergence(alphas, target_concentration=1.0):
    """
    Compute KL divergence between Dirichlet distributions.
    KL(Dir(alpha) || Dir(alpha_0)) where alpha_0 is uniform prior
    """
    batch_size = alphas.size(0)
    num_classes = alphas.size(1)
    
    # Target uniform Dirichlet parameters
    target_alphas = torch.ones_like(alphas) * target_concentration
    
    # Sum of parameters
    alpha_sum = torch.sum(alphas, dim=1, keepdim=True)
    target_sum = torch.sum(target_alphas, dim=1, keepdim=True)
    
    # Log gamma functions
    # Using torch.lgamma for numerical stability
    kl_div = torch.lgamma(alpha_sum) - torch.lgamma(target_sum)
    kl_div -= torch.sum(torch.lgamma(alphas) - torch.lgamma(target_alphas), dim=1, keepdim=True)
    kl_div += torch.sum((alphas - target_alphas) * (torch.digamma(alphas) - torch.digamma(alpha_sum)), dim=1, keepdim=True)
    
    return kl_div.squeeze()


def evidential_loss(evidence, targets, epoch, annealing_coefficient=1.0, regularization_coefficient=0.5):
    """
    Complete evidential loss function with KL regularization
    
    Args:
        evidence: Evidence values from model (batch_size, num_classes)
        targets: Ground truth labels (batch_size,)
        epoch: Current epoch for annealing
        annealing_coefficient: Coefficient for KL annealing
        regularization_coefficient: Weight for KL regularization term
    """
    batch_size = evidence.size(0)
    num_classes = evidence.size(1)
    
    # Convert evidence to Dirichlet parameters
    alphas = evidence + 1.0
    alpha_sum = torch.sum(alphas, dim=1, keepdim=True)
    
    # One-hot encode targets
    targets_one_hot = F.one_hot(targets, num_classes=num_classes).float()
    
    # Expected log-likelihood (first term)
    expected_log_likelihood = -torch.sum(targets_one_hot * (torch.digamma(alphas) - torch.digamma(alpha_sum)), dim=1)
    
    # KL divergence regularization (second term)
    # KL divergence from uniform prior
    kl_div = dirichlet_kl_divergence(alphas, target_concentration=1.0)
    
    # Annealing factor for KL term (starts small, increases over time)
    annealing_factor = min(1.0, annealing_coefficient * epoch / 100.0)
    
    # Total loss: negative expected log-likelihood + regularized KL divergence
    loss = expected_log_likelihood + annealing_factor * regularization_coefficient * kl_div
    
    # Additional evidence regularization to prevent overconfidence
    # Penalize very high evidence values that are incorrect
    incorrect_evidence = torch.sum(evidence * (1 - targets_one_hot), dim=1)
    evidence_penalty = torch.mean(F.relu(incorrect_evidence - 2.0))
    
    total_loss = torch.mean(loss) + 0.005 * evidence_penalty
    
    return total_loss, torch.mean(-expected_log_likelihood), torch.mean(kl_div), evidence_penalty


def train_evidential_classifier(model, dataloader, optimizer, device, epoch, class_weights=None):
    """
    Training function for full evidential learning
    """
    model.train()
    total_loss = 0.0
    total_nll = 0.0
    total_kl = 0.0
    total_penalty = 0.0
    correct = 0
    total = 0
    
    for data in tqdm.tqdm(dataloader):
        optimizer.zero_grad()

        X = data[0].to(device, dtype=torch.float)
        labels = data[1].to(device, dtype=torch.int).long()
        X = X.view(X.size(0), 1, X.size(1))
        
        # Forward pass
        evidence, _ = model(X)
        evidence = evidence.squeeze(0)  # Remove sequence dim
        
        # Compute evidential loss
        loss, nll, kl_div, penalty = evidential_loss(
            evidence, labels, epoch, 
            annealing_coefficient=1.0, 
            regularization_coefficient=0.5
        )
        
        # Apply class weights by adjusting the loss
        if class_weights is not None:
            # Weight adjustment based on class distribution
            class_loss_weights = class_weights[labels]
            weighted_loss = loss * torch.mean(class_loss_weights)
            loss = weighted_loss
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        # Calculate accuracy based on expected probabilities
        alphas = evidence + 1.0
        alpha_sum = torch.sum(alphas, dim=1, keepdim=True)
        prob = alphas / alpha_sum
        predicted = torch.argmax(prob, dim=1)
        
        total_loss += loss.item()
        total_nll += nll.item()
        total_kl += kl_div.item()
        total_penalty += penalty.item()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    avg_nll = total_nll / len(dataloader)
    avg_kl = total_kl / len(dataloader)
    avg_penalty = total_penalty / len(dataloader)
    accuracy = 100.0 * correct / total
    
    return avg_loss, avg_nll, avg_kl, avg_penalty, accuracy


def evaluate_evidential_classifier(model, test_loader, device):
    """
    Evaluate evidential classifier with uncertainty analysis
    """
    model.eval()
    correct = 0
    total = 0
    
    all_predictions = []
    all_uncertainties = []
    all_epistemic = []
    all_aleatoric = []
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
            all_uncertainties.append(total_unc.cpu())
            all_epistemic.append(epistemic.cpu())
            all_aleatoric.append(aleatoric.cpu())
            all_confidences.append(confidence.cpu())
            all_targets.append(labels.cpu())
            all_alphas.append(alpha_sum.cpu())
    
    accuracy = 100.0 * correct / total
    
    predictions = torch.cat(all_predictions, dim=1)
    uncertainties = torch.cat(all_uncertainties, dim=1)
    epistemic_unc = torch.cat(all_epistemic, dim=1)
    aleatoric_unc = torch.cat(all_aleatoric, dim=1)
    confidences = torch.cat(all_confidences, dim=0)
    targets = torch.cat(all_targets, dim=0)
    alphas = torch.cat(all_alphas, dim=1)
    
    return (accuracy, predictions, uncertainties, epistemic_unc, 
            aleatoric_unc, confidences, targets, alphas)


def create_model(input_length=860, num_classes=2):
    """Factory function to create the model with proper initialization"""
    model = ImprovedEvidentialIENet(
        num_classes=num_classes,
        input_length=input_length,
        dropout=0.1
    )
    
    # Initialize weights with Xavier/Kaiming
    for m in model.modules():
        if isinstance(m, nn.Conv1d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    
    return model


if __name__ == '__main__':
    X_path = ['data/X_train_860.npy']
    y_path = ['data/y_train.npy']

    epochs = 100
    model_name = 'evidential_transformer_v5'
    batch_size = 128
    learning_rate = 0.0001 # Slightly lower LR for more stable training
    num_classes = 2
    validation_split = 0.3  # 28% for validation
    
    dataset = ImpactEchoDatasetClassifierAug(X_path, y_path=y_path, array_size=860)
    print(f"Total number of samples: {len(dataset)}")
    
    # Split dataset into train and validation
    val_size = int(len(dataset) * validation_split)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    train_dataloader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Calculate class weights for imbalanced data
    y_data = np.load(y_path[0])
    y_data[y_data < 1] = 0
    y_data[y_data > 0] = 1
    class_counts = np.bincount(y_data.astype(int))
    total_samples = len(y_data)
    class_weights = torch.FloatTensor([total_samples / (2 * count) for count in class_counts]).to(device)
    print(f"Class distribution: {class_counts}")
    print(f"Class weights: {class_weights}")

    # Initialize model
    model = create_model().to(device=device)
    
    # Use AdamW with weight decay for better regularization
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    print('Training Full Evidential Deep Learning classifier...')
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # MODIFIED: Track best validation accuracy instead of loss
    best_val_accuracy = 0.0  # Start from 0 for accuracy
    best_val_loss = float('inf')  # Still track loss for logging
    patience = 25
    patience_counter = 0
    
    # Track validation metrics history
    val_accuracy_history = []
    val_loss_history = []
    
    for epoch in range(epochs):        
        # Train
        train_loss, train_nll, train_kl_div, train_penalty, train_acc = train_evidential_classifier(
            model, train_dataloader, optimizer, device, epoch, class_weights=class_weights
        )
        
        # Validation
        model.eval()
        val_total_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for data in val_dataloader:
                X = data[0].to(device, dtype=torch.float)
                labels = data[1].to(device, dtype=torch.int).long()
                X = X.view(X.size(0), 1, X.size(1))
                
                evidence, _ = model(X)
                evidence = evidence.squeeze(0)
                
                val_loss, _, _, _ = evidential_loss(
                    evidence, labels, epoch,
                    annealing_coefficient=1.0,
                    regularization_coefficient=0.5
                )
                
                # Calculate validation accuracy
                alphas = evidence + 1.0
                alpha_sum = torch.sum(alphas, dim=1, keepdim=True)
                prob = alphas / alpha_sum
                predicted = torch.argmax(prob, dim=1)
                
                val_total_loss += val_loss.item()
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        avg_val_loss = val_total_loss / len(val_dataloader)
        val_accuracy = 100.0 * val_correct / val_total
        
        # Store history
        val_accuracy_history.append(val_accuracy)
        val_loss_history.append(avg_val_loss)
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch:03d}, Train Loss: {train_loss:.4f} (NLL: {train_nll:.4f}, KL: {train_kl_div:.4f}, Penalty: {train_penalty:.4f}), '
              f'Train Acc: {train_acc:.2f}%, Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%, LR: {current_lr:.8f}')
        
        # Log to tensorboard
        writer.add_scalar("Loss/train_total", train_loss, epoch)
        writer.add_scalar("Loss/train_nll", train_nll, epoch)
        writer.add_scalar("Loss/train_kl_divergence", train_kl_div, epoch)
        writer.add_scalar("Loss/train_evidence_penalty", train_penalty, epoch)
        writer.add_scalar("Loss/val_total", avg_val_loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Accuracy/val", val_accuracy, epoch)
        writer.add_scalar("Learning_Rate", current_lr, epoch)
        
        # MODIFIED: Save best model based on validation ACCURACY
        if val_accuracy > best_val_accuracy:
            print(f"🎯 New Best Validation Accuracy: {val_accuracy:.2f}% (was {best_val_accuracy:.2f}%), Val Loss: {avg_val_loss:.4f}")
            best_val_accuracy = val_accuracy
            best_val_loss = avg_val_loss  # Track the loss at best accuracy
            patience_counter = 0
            save_model(model, f'weights/{model_name}_best_acc.pth')
            
            # Save comprehensive checkpoint info
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_accuracy': best_val_accuracy,
                'best_val_loss_at_best_acc': best_val_loss,
                'train_loss': train_loss,
                'val_loss': avg_val_loss,
                'train_accuracy': train_acc,
                'val_accuracy': val_accuracy,
                'val_accuracy_history': val_accuracy_history,
                'val_loss_history': val_loss_history
            }, f'weights/{model_name}_checkpoint_best_acc.pth')
            
            # Also save a backup with epoch number
            torch.save(model.state_dict(), f'weights/{model_name}_epoch_{epoch}_acc_{val_accuracy:.1f}.pth')
        else:
            patience_counter += 1
            
        # Optional: Also save if we achieve best loss (for comparison)
        if avg_val_loss < best_val_loss and avg_val_loss != best_val_loss:
            print(f"📉 Best Validation Loss: {avg_val_loss:.4f}, but accuracy is {val_accuracy:.2f}% (best acc: {best_val_accuracy:.2f}%)")
            # You can optionally save this model too with a different name
            # save_model(model, f'weights/{model_name}_best_loss.pth')
        
        # Early stopping based on accuracy
        if patience_counter >= patience:
            print(f"Early stopping triggered after {patience} epochs without accuracy improvement")
            print(f"Best validation accuracy: {best_val_accuracy:.2f}% (with loss: {best_val_loss:.4f})")
            break
        
        # Additional stopping criterion: if accuracy is very high
        if val_accuracy >= 99.5:
            print(f"Stopping early due to near-perfect validation accuracy: {val_accuracy:.2f}%")
            break
        
        scheduler.step()

    print("\n" + "="*50)
    print(f"Training completed!")
    print(f"Best validation accuracy achieved: {best_val_accuracy:.2f}%")
    print(f"Validation loss at best accuracy: {best_val_loss:.4f}")
    print("="*50 + "\n")

    # Load the best model for final evaluation
    print("Loading best model for final evaluation...")
    checkpoint = torch.load(f'weights/{model_name}_checkpoint_best_acc.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from epoch {checkpoint['epoch']} with val accuracy {checkpoint['best_val_accuracy']:.2f}%")

    # Test evaluation
    print("\nEvaluating model on test set...")
    
    # Load test data
    X_test_path = ['data/X_test_860.npy']
    y_test_path = ['data/y_test.npy']
    test_dataset = ImpactEchoDatasetClassifier(X_test_path, y_path=y_test_path, array_size=860)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Evidential evaluation
    (accuracy, predictions, uncertainties, epistemic_unc, 
     aleatoric_unc, confidences, targets, alphas) = evaluate_evidential_classifier(model, test_loader, device)
    
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Mean Total Uncertainty: {uncertainties.mean():.4f} ± {uncertainties.std():.4f}")
    print(f"Mean Epistemic Uncertainty: {epistemic_unc.mean():.4f} ± {epistemic_unc.std():.4f}")
    print(f"Mean Aleatoric Uncertainty: {aleatoric_unc.mean():.4f} ± {aleatoric_unc.std():.4f}")
    print(f"Mean Confidence: {confidences.mean():.4f} ± {confidences.std():.4f}")
    print(f"Mean Alpha Sum: {alphas.mean():.4f} ± {alphas.std():.4f}")
    
    # Analyze uncertainty by correctness
    pred_classes = torch.argmax(predictions.squeeze(0), dim=1)
    correct_preds = (pred_classes == targets)
    
    print(f"\nUncertainty Analysis:")
    print(f"Correct Predictions - Mean Total Uncertainty: {uncertainties.squeeze(0)[correct_preds].mean():.4f}")
    print(f"Incorrect Predictions - Mean Total Uncertainty: {uncertainties.squeeze(0)[~correct_preds].mean():.4f}")
    print(f"Correct Predictions - Mean Epistemic Uncertainty: {epistemic_unc.squeeze(0)[correct_preds].mean():.4f}")
    print(f"Incorrect Predictions - Mean Epistemic Uncertainty: {epistemic_unc.squeeze(0)[~correct_preds].mean():.4f}")
    print(f"Correct Predictions - Mean Confidence: {confidences[correct_preds].mean():.4f}")
    print(f"Incorrect Predictions - Mean Confidence: {confidences[~correct_preds].mean():.4f}")
    
    # Calculate per-class accuracy
    print(f"\nPer-class Performance:")
    for class_idx in range(num_classes):
        class_mask = targets == class_idx
        class_correct = (pred_classes[class_mask] == targets[class_mask]).sum().item()
        class_total = class_mask.sum().item()
        class_acc = 100.0 * class_correct / class_total if class_total > 0 else 0
        print(f"Class {class_idx}: {class_acc:.2f}% ({class_correct}/{class_total})")
    
    # Save comprehensive results
    torch.save({
        'predictions': predictions,
        'total_uncertainties': uncertainties,
        'epistemic_uncertainties': epistemic_unc,
        'aleatoric_uncertainties': aleatoric_unc,
        'confidences': confidences,
        'targets': targets,
        'alphas': alphas,
        'accuracy': accuracy,
        'best_val_accuracy': best_val_accuracy,
        'best_val_loss_at_best_acc': best_val_loss,
        'model_config': {
            'num_classes': num_classes,
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'class_weights': class_weights.cpu(),
            'model_type': 'full_evidential'
        }
    }, f'weights/{model_name}_results.pth')
    
    print(f"\nResults saved to weights/{model_name}_results.pth")

    writer.flush()
    writer.close()