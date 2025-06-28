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


class SimpleEvidentialIENet(nn.Module):
    """
    Simplified Evidential IENet - starts as regular classifier, gradually becomes evidential
    """
    
    def __init__(self, num_classes=2, verbose=False):
        super(SimpleEvidentialIENet, self).__init__()
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
        
        # Single output layer that can be interpreted as either logits or evidence
        self.output_layer = nn.Linear(in_features=64, out_features=num_classes)

    def forward(self, x, use_evidential=False):
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
        
        # Output layer
        output = self.output_layer(features)
        
        if use_evidential:
            # Interpret as evidence (must be positive)
            evidence = F.softplus(output)
            alphas = evidence + 1.0
            return alphas, evidence, features
        else:
            # Interpret as logits for standard classification
            return output, features

    def predict_with_uncertainty(self, x):
        """
        Get predictions with evidential uncertainty
        """
        with torch.no_grad():
            alphas, evidence, _ = self.forward(x, use_evidential=True)
            
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


def train_classifier(model, dataloader, optimizer, criterion, device, epoch, use_evidential=False, class_weights=None):
    """
    Unified training function for both standard and evidential modes
    """
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for data in tqdm.tqdm(dataloader):
        optimizer.zero_grad()

        X = data[0].to(device, dtype=torch.float)
        labels = data[1].to(device, dtype=torch.int).long()
        X = X.view(X.size(0), 1, X.size(1))
        
        if use_evidential:
            # Evidential mode
            alphas, evidence, _ = model(X, use_evidential=True)
            alphas = alphas.squeeze(0)  # Remove sequence dim
            
            # Use Dirichlet-based loss
            alpha_sum = torch.sum(alphas, dim=1, keepdim=True)
            prob = alphas / alpha_sum
            
            # Cross-entropy loss on expected probabilities
            loss = F.cross_entropy(torch.log(prob + 1e-8), labels, weight=class_weights)
            
            # Add evidence regularization (encourage higher evidence for training samples)
            evidence_reg = torch.mean(torch.sum(F.relu(3.0 - evidence.squeeze(0)), dim=1))
            loss = loss + 0.01 * evidence_reg
            
            # Calculate accuracy
            predicted = torch.argmax(prob, dim=1)
        else:
            # Standard classification mode
            logits, _ = model(X, use_evidential=False)
            logits = logits.squeeze(0)  # Remove sequence dim
            
            # Standard cross-entropy loss
            loss = F.cross_entropy(logits, labels, weight=class_weights)
            
            # Calculate accuracy
            predicted = torch.argmax(logits, dim=1)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(dataloader.dataset)
    accuracy = 100.0 * correct / total
    
    return avg_loss, accuracy


def evaluate_classifier(model, test_loader, device, use_evidential=False):
    """
    Evaluate classifier in either standard or evidential mode
    """
    model.eval()
    correct = 0
    total = 0
    
    all_predictions = []
    all_uncertainties = []
    all_targets = []
    
    with torch.no_grad():
        for data in test_loader:
            X = data[0].to(device, dtype=torch.float)
            labels = data[1].to(device, dtype=torch.int).long()
            X = X.view(X.size(0), 1, X.size(1))
            
            if use_evidential:
                prob, epistemic, aleatoric, total_unc, confidence, alpha_sum = model.predict_with_uncertainty(X)
                predicted = torch.argmax(prob.squeeze(0), dim=1)
                
                all_predictions.append(prob.cpu())
                all_uncertainties.append(total_unc.cpu())
            else:
                logits, _ = model(X, use_evidential=False)
                predicted = torch.argmax(logits.squeeze(0), dim=1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_targets.append(labels.cpu())
    
    accuracy = 100.0 * correct / total
    
    if use_evidential:
        predictions = torch.cat(all_predictions, dim=1)
        uncertainties = torch.cat(all_uncertainties, dim=1)
        targets = torch.cat(all_targets, dim=0)
        return accuracy, predictions, uncertainties, targets
    else:
        targets = torch.cat(all_targets, dim=0)
        return accuracy, None, None, targets


if __name__ == '__main__':
    X_path = ['data/X_train_860.npy']
    y_path = ['data/y_train.npy']

    epochs = 200
    model_name = 'evidential_simple_model'
    batch_size = 32
    learning_rate = 0.001
    num_classes = 2
    
    # Switch to evidential mode after this many epochs
    evidential_switch_epoch = 80
    
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
    model = SimpleEvidentialIENet(num_classes=num_classes, verbose=False).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=15, factor=0.7)

    print('Training Evidential Deep Learning classifier (progressive)...')
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Standard training: epochs 0-{evidential_switch_epoch-1}")
    print(f"Evidential training: epochs {evidential_switch_epoch}-{epochs-1}")

    best_loss = float('inf')
    best_accuracy = 0.0
    
    for epoch in range(epochs):
        # Decide which mode to use
        use_evidential = epoch >= evidential_switch_epoch
        mode_str = "Evidential" if use_evidential else "Standard"
        
        # Train
        loss, train_acc = train_classifier(
            model, dataloader, optimizer, None, device, epoch, 
            use_evidential=use_evidential, class_weights=class_weights
        )
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f'Epoch {epoch:03d} ({mode_str}), Loss: {loss:.4f}, TrainAcc: {train_acc:.2f}%, LR: {current_lr:.6f}')
        
        # Log to tensorboard
        writer.add_scalar("Loss/total", loss, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Learning_Rate", current_lr, epoch)
        writer.add_scalar("Mode", 1 if use_evidential else 0, epoch)
        
        # Save best model based on accuracy (more important than loss for imbalanced data)
        if train_acc > best_accuracy:
            print(f"Best Train Accuracy: {train_acc:.2f}% (was {best_accuracy:.2f}%)")
            best_accuracy = train_acc
            save_model(model, f'weights/{model_name}.pth')
        
        scheduler.step(loss)
        
        # Lower learning rate when switching to evidential mode
        if epoch == evidential_switch_epoch:
            for param_group in optimizer.param_groups:
                param_group['lr'] *= 0.5
            print(f"Switched to evidential mode, reduced LR to {optimizer.param_groups[0]['lr']:.6f}")

    # Test evaluation
    print("\nEvaluating model...")
    
    # Load test data
    X_test_path = ['data/X_test_860.npy']
    y_test_path = ['data/y_test.npy']
    test_dataset = ImpactEchoDatasetClassifier(X_test_path, y_path=y_test_path, array_size=860)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # Standard evaluation
    standard_accuracy, _, _, targets = evaluate_classifier(model, test_loader, device, use_evidential=False)
    print(f"Test Accuracy (Standard Mode): {standard_accuracy:.2f}%")
    
    # Evidential evaluation
    evidential_accuracy, predictions, uncertainties, _ = evaluate_classifier(model, test_loader, device, use_evidential=True)
    print(f"Test Accuracy (Evidential Mode): {evidential_accuracy:.2f}%")
    
    if uncertainties is not None:
        print(f"Mean Uncertainty: {uncertainties.mean():.4f} ± {uncertainties.std():.4f}")
        
        # Analyze uncertainty
        pred_classes = torch.argmax(predictions.squeeze(0), dim=1)
        correct_preds = (pred_classes == targets)
        
        print(f"Correct Predictions - Mean Uncertainty: {uncertainties.squeeze(0)[correct_preds].mean():.4f}")
        print(f"Incorrect Predictions - Mean Uncertainty: {uncertainties.squeeze(0)[~correct_preds].mean():.4f}")
    
    # Save results
    torch.save({
        'predictions': predictions,
        'uncertainties': uncertainties,
        'targets': targets,
        'standard_accuracy': standard_accuracy,
        'evidential_accuracy': evidential_accuracy,
        'model_config': {
            'num_classes': num_classes,
            'evidential_switch_epoch': evidential_switch_epoch,
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'class_weights': class_weights.cpu()
        }
    }, f'weights/{model_name}_results.pth')
    
    print(f"Results saved to weights/{model_name}_results.pth")

    writer.flush()
    writer.close()