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
from torch.utils.data import DataLoader
from dataloaders.dataloader import ImpactEchoDatasetClassifier
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


class FullEvidentialIENet(nn.Module):
    """
    Full Evidential IENet with complete Dirichlet-based loss including KL regularization
    """
    
    def __init__(self, num_classes=2, verbose=False):
        super(FullEvidentialIENet, self).__init__()
        self.verbose = verbose
        self.num_classes = num_classes
        
        # Feature extraction layers (same as baseline)
        self.residual_1 = ResidualBlock(1, 8, 200)
        self.residual_2 = ResidualBlock(8, 16, 100)
        self.residual_3 = ResidualBlock(16, 16, 50)
        self.residual_4 = ResidualBlock(16, 32, 25)
        self.residual_5 = ResidualBlock(32, 64, 13)
        self.residual_6 = ResidualBlock(64, 64, 7)
        
        # LSTM layers
        self.bilstm_1 = nn.LSTM(input_size=832, hidden_size=32, num_layers=1, 
                               batch_first=True, bidirectional=True)
        self.bilstm_2 = nn.LSTM(input_size=64, hidden_size=32, num_layers=1, 
                               batch_first=True, bidirectional=True)
        self.bilstm_3 = nn.LSTM(input_size=64, hidden_size=32, num_layers=1, 
                               batch_first=True, bidirectional=True)
        
        # Evidence output layer
        self.evidence_layer = nn.Linear(in_features=64, out_features=num_classes)

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

        # LSTM layers
        x, _ = self.bilstm_1(x)
        x, _ = self.bilstm_2(x)
        features, _ = self.bilstm_3(x)
        
        # Evidence output (must be positive)
        evidence = F.softplus(self.evidence_layer(features))
        
        return evidence, features

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
            # 1. Epistemic uncertainty (vacuity): K / S 
            epistemic_uncertainty = self.num_classes / alpha_sum
            
            # 2. Aleatoric uncertainty (expected data uncertainty)
            aleatoric_uncertainty = torch.sum(prob * (1 - prob) / (alpha_sum + 1), dim=-1, keepdim=True)
            
            # 3. Total uncertainty
            total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
            
            # 4. Confidence (max probability)
            confidence = torch.max(prob, dim=-1)[0]
            
            return prob, epistemic_uncertainty, aleatoric_uncertainty, total_uncertainty, confidence, alpha_sum


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


def evidential_loss(evidence, targets, epoch, annealing_coefficient=1.0, regularization_coefficient=0.01):
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
    expected_log_likelihood = torch.sum(targets_one_hot * (torch.digamma(alphas) - torch.digamma(alpha_sum)), dim=1)
    
    # KL divergence regularization (second term)
    # KL divergence from uniform prior
    kl_div = dirichlet_kl_divergence(alphas, target_concentration=1.0)
    
    # Annealing factor for KL term (starts small, increases over time)
    annealing_factor = min(1.0, annealing_coefficient * epoch / 100.0)
    
    # Total loss: negative expected log-likelihood + regularized KL divergence
    loss = -expected_log_likelihood + annealing_factor * regularization_coefficient * kl_div
    
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
            regularization_coefficient=0.01
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


if __name__ == '__main__':
    X_path = ['data/X_train_860.npy']
    y_path = ['data/y_train.npy']

    epochs = 250
    model_name = 'evidential_full_v3'
    batch_size = 32
    learning_rate = 0.0005  # Slightly lower LR for more stable training
    num_classes = 2
    
    dataset = ImpactEchoDatasetClassifier(X_path, y_path=y_path, array_size=860)
    print(f"Total number of training samples: {len(dataset)}")
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)

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
    model = FullEvidentialIENet(num_classes=num_classes, verbose=False).to(device)
    
    # Use AdamW with weight decay for better regularization
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    print('Training Full Evidential Deep Learning classifier...')
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    best_loss = float('inf')
    best_accuracy = 0.0
    
    for epoch in range(epochs):        
        # Train
        loss, nll, kl_div, penalty, train_acc = train_evidential_classifier(
            model, dataloader, optimizer, device, epoch, class_weights=class_weights
        )
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch:03d}, Loss: {loss:.4f} (NLL: {nll:.4f}, KL: {kl_div:.4f}, Penalty: {penalty:.4f}), '
              f'TrainAcc: {train_acc:.2f}%, LR: {current_lr:.8f}')
        
        # Log to tensorboard
        writer.add_scalar("Loss/total", loss, epoch)
        writer.add_scalar("Loss/nll", nll, epoch)
        writer.add_scalar("Loss/kl_divergence", kl_div, epoch)
        writer.add_scalar("Loss/evidence_penalty", penalty, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Learning_Rate", current_lr, epoch)
        
        # Save best model based on accuracy
        if train_acc > best_accuracy:
            print(f"Best Train Accuracy: {train_acc:.2f}% (was {best_accuracy:.2f}%)")
            best_accuracy = train_acc
            save_model(model, f'weights/{model_name}.pth')
        
        scheduler.step()

    # Test evaluation
    print("\nEvaluating model...")
    
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
        'model_config': {
            'num_classes': num_classes,
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'class_weights': class_weights.cpu(),
            'model_type': 'full_evidential'
        }
    }, f'weights/{model_name}_results.pth')
    
    print(f"Results saved to weights/{model_name}_results.pth")

    writer.flush()
    writer.close()