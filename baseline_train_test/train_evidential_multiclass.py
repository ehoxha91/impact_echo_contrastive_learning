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


class MulticlassEvidentialIENet(nn.Module):
    """
    Multi-class Evidential IENet for 5-class defect classification
    4 defect types + 1 solid concrete (no-defect)
    """
    
    def __init__(self, num_classes=5, verbose=False):
        super(MulticlassEvidentialIENet, self).__init__()
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
        
        # Evidence output layer for 5 classes
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
        Get predictions with evidential uncertainty measures for multi-class
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


def evidential_multiclass_loss(evidence, targets, epoch, annealing_coefficient=1.0, regularization_coefficient=0.01):
    """
    Multi-class evidential loss function with KL regularization
    
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
    
    total_loss = torch.mean(loss) + 0.01 * evidence_penalty
    
    return total_loss, torch.mean(-expected_log_likelihood), torch.mean(kl_div), evidence_penalty


class MulticlassImpactEchoDataset(ImpactEchoDatasetClassifier):
    """
    Modified dataset class that preserves multi-class labels instead of binarizing them
    """
    
    def __init__(self, X_path, y_path, array_size=860, shuffle=False):
        # Initialize base class but override label processing
        self.sr = [200000, 104200]
        self.X = np.array([])
        self.array_size = array_size
        self.shuffle = shuffle
        
        # Load X data
        items = 0
        for path in X_path:
            tmp = np.load(path)
            tmp = tmp[:, 0:self.array_size]
            items += tmp.shape[0]
            self.X = np.append(self.X, tmp)
        self.X = np.reshape(self.X, (items, -1))
        
        # Load y data - PRESERVE ORIGINAL LABELS (don't binarize)
        self.labels1 = np.load(y_path[0])
        print(f"Loaded dataset {y_path[0]}, with {len(self.labels1)} data points")
        
        # Map labels to ensure they're in range [0, 4] for 5 classes
        # Class 0: solid concrete (no defect)
        # Classes 1-4: different defect types
        self.labels1 = np.clip(self.labels1, 0, 4).astype(int)
        print(f"Multi-class label distribution: {np.bincount(self.labels1)}")
        
        self.dataset1_size = len(self.labels1)
        self.y = self.labels1
        
        if self.shuffle:
            self.X_y = np.column_stack((self.X, self.y))
            np.random.shuffle(self.X_y)
            self.y = [int(xa) for xa in self.X_y[:,-1]]
            self.X = self.X_y[:,:-1]


def train_multiclass_evidential_classifier(model, dataloader, optimizer, device, epoch, class_weights=None):
    """
    Training function for multi-class evidential learning
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
        loss, nll, kl_div, penalty = evidential_multiclass_loss(
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


def evaluate_multiclass_evidential_classifier(model, test_loader, device):
    """
    Evaluate multi-class evidential classifier with uncertainty analysis
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

    epochs = 100
    model_name = 'evidential_multiclass_v1'
    batch_size = 16
    learning_rate = 0.0001
    num_classes = 5  # 4 defect types + 1 solid concrete
    validation_split = 0.28
    
    # Use multi-class dataset
    dataset = MulticlassImpactEchoDataset(X_path, y_path=y_path, array_size=860)
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
    
    # Calculate class weights for imbalanced data (5 classes)
    y_data = np.load(y_path[0])
    y_data = np.clip(y_data, 0, 4).astype(int)  # Ensure 5 classes [0,1,2,3,4]
    class_counts = np.bincount(y_data, minlength=5)
    total_samples = len(y_data)
    class_weights = torch.FloatTensor([total_samples / (num_classes * count) if count > 0 else 0.0 for count in class_counts]).to(device)
    print(f"Class distribution: {class_counts}")
    print(f"Class weights: {class_weights}")

    # Initialize multi-class model
    model = MulticlassEvidentialIENet(num_classes=num_classes, verbose=False).to(device)
    
    # Use AdamW with weight decay for better regularization
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)

    print('Training Multi-class Evidential Deep Learning classifier...')
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    best_val_loss = float('inf')
    best_val_accuracy = 0.0
    patience = 15
    patience_counter = 0
    
    for epoch in range(epochs):        
        # Train
        train_loss, train_nll, train_kl_div, train_penalty, train_acc = train_multiclass_evidential_classifier(
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
                
                val_loss, _, _, _ = evidential_multiclass_loss(
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
        
        # Save best model based on validation loss
        if avg_val_loss < best_val_loss:
            print(f"Best Validation Loss: {avg_val_loss:.4f} (was {best_val_loss:.4f}), Val Acc: {val_accuracy:.2f}%")
            best_val_loss = avg_val_loss
            best_val_accuracy = val_accuracy
            patience_counter = 0
            save_model(model, f'weights/{model_name}.pth')
            
            # Save additional checkpoint info
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_val_loss': best_val_loss,
                'best_val_accuracy': best_val_accuracy,
                'train_loss': train_loss,
                'val_loss': avg_val_loss,
                'train_accuracy': train_acc,
                'val_accuracy': val_accuracy
            }, f'weights/{model_name}_checkpoint.pth')
        else:
            patience_counter += 1
        
        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {patience} epochs without improvement")
            print(f"Best validation loss: {best_val_loss:.4f}, Best validation accuracy: {best_val_accuracy:.2f}%")
            break
        
        scheduler.step()

    # Test evaluation
    print("\nEvaluating multi-class model...")
    
    # Load test data
    X_test_path = ['data/X_test_860.npy']
    y_test_path = ['data/y_test.npy']
    test_dataset = MulticlassImpactEchoDataset(X_test_path, y_path=y_test_path, array_size=860)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Multi-class evidential evaluation
    (accuracy, predictions, uncertainties, epistemic_unc, 
     aleatoric_unc, confidences, targets, alphas) = evaluate_multiclass_evidential_classifier(model, test_loader, device)
    
    print(f"Test Accuracy: {accuracy:.2f}%")
    print(f"Mean Total Uncertainty: {uncertainties.mean():.4f} ± {uncertainties.std():.4f}")
    print(f"Mean Epistemic Uncertainty: {epistemic_unc.mean():.4f} ± {epistemic_unc.std():.4f}")
    print(f"Mean Aleatoric Uncertainty: {aleatoric_unc.mean():.4f} ± {aleatoric_unc.std():.4f}")
    print(f"Mean Confidence: {confidences.mean():.4f} ± {confidences.std():.4f}")
    print(f"Mean Alpha Sum: {alphas.mean():.4f} ± {alphas.std():.4f}")
    
    # Multi-class specific analysis
    pred_classes = torch.argmax(predictions.squeeze(0), dim=1)
    correct_preds = (pred_classes == targets)
    
    print(f"\nMulti-class Analysis:")
    print(f"Correct Predictions - Mean Total Uncertainty: {uncertainties.squeeze(0)[correct_preds].mean():.4f}")
    print(f"Incorrect Predictions - Mean Total Uncertainty: {uncertainties.squeeze(0)[~correct_preds].mean():.4f}")
    
    # Class-wise accuracy
    for class_idx in range(num_classes):
        class_mask = targets == class_idx
        if class_mask.sum() > 0:
            class_accuracy = (pred_classes[class_mask] == class_idx).float().mean()
            print(f"Class {class_idx} Accuracy: {class_accuracy*100:.2f}% ({class_mask.sum()} samples)")
    
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
            'model_type': 'multiclass_evidential'
        }
    }, f'weights/{model_name}_results.pth')
    
    print(f"Results saved to weights/{model_name}_results.pth")

    writer.flush()
    writer.close()