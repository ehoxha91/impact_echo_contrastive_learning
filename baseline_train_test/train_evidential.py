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


class EvidentialIENet(nn.Module):
    """
    IENet with Evidential Deep Learning for uncertainty quantification.
    Outputs Dirichlet parameters (alphas) for each class.
    """
    
    def __init__(self, num_classes=2, verbose=False):
        super(EvidentialIENet, self).__init__()
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
        
        # Evidential output layer - outputs evidence for each class
        self.evidential_layer = nn.Linear(in_features=64, out_features=num_classes)

    def forward(self, x):
        # Feature extraction
        x = self.residual_1(x)
        if self.verbose:
            print(f"residual_1: {x.shape}")

        x = self.residual_2(x)
        if self.verbose:
            print(f"residual_2: {x.shape}")

        x = self.residual_3(x)
        if self.verbose:
            print(f"residual_3: {x.shape}")

        x = self.residual_4(x)
        if self.verbose:
            print(f"residual_4: {x.shape}")

        x = self.residual_5(x)
        if self.verbose:
            print(f"residual_5: {x.shape}")

        x = self.residual_6(x)
        if self.verbose:
            print(f"residual_6: {x.shape}")

        # Reshape for LSTM input
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        if self.verbose:
            print(f"flatten: {x.shape}")

        # LSTM layers - add sequence dimension
        x = x.unsqueeze(1)  # [batch_size, 1, features]
        x, _ = self.bilstm_1(x)
        if self.verbose:
            print(f"bilstm_1: {x.shape}")

        x, _ = self.bilstm_2(x)
        if self.verbose:
            print(f"bilstm_2: {x.shape}")

        features, _ = self.bilstm_3(x)
        # Remove sequence dimension: [batch_size, 1, features] -> [batch_size, features]
        features = features.squeeze(1)
        if self.verbose:
            print(f"bilstm_3: {features.shape}")
        
        # Get evidence (must be positive)
        evidence = F.softplus(self.evidential_layer(features))  # Smooth positive activation
        
        # Convert evidence to Dirichlet parameters (alphas)
        alphas = evidence + 1.0  # alpha_k = e_k + 1
        
        if self.verbose:
            print(f"evidence: {evidence.shape}")
            print(f"alphas: {alphas.shape}")
        
        return alphas, evidence, features

    def predict_with_uncertainty(self, x):
        """
        Get predictions with evidential uncertainty
        """
        with torch.no_grad():
            alphas, evidence, _ = self.forward(x)
            
            # Sum of alphas (strength of Dirichlet)
            alpha_sum = torch.sum(alphas, dim=-1, keepdim=True)
            
            # Expected probabilities (mean of Dirichlet)
            prob = alphas / alpha_sum
            
            # Uncertainty measures
            # 1. Epistemic uncertainty (vacuity): K / S where K=num_classes, S=sum of alphas
            epistemic_uncertainty = self.num_classes / alpha_sum
            
            # 2. Aleatoric uncertainty (expected data uncertainty)
            aleatoric_uncertainty = torch.sum(prob * (1 - prob) / (alpha_sum + 1), dim=-1, keepdim=True)
            
            # 3. Total uncertainty
            total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
            
            # 4. Confidence (max probability)
            confidence = torch.max(prob, dim=-1)[0]
            
            return prob, epistemic_uncertainty, aleatoric_uncertainty, total_uncertainty, confidence, alpha_sum


class EvidentialLoss(nn.Module):
    """
    Evidential Deep Learning Loss Function with warm-up and class weighting
    """
    
    def __init__(self, num_classes=2, annealing_coeff=1.0, regularization_coeff=0.01, warmup_epochs=20, 
                 annealing_epochs=50, class_weights=None):
        super(EvidentialLoss, self).__init__()
        self.num_classes = num_classes
        self.annealing_coeff = annealing_coeff
        self.regularization_coeff = regularization_coeff
        self.warmup_epochs = warmup_epochs
        self.annealing_epochs = annealing_epochs
        self.class_weights = class_weights
        
    def forward(self, alphas, targets, epoch=0):
        """
        Progressive evidential loss with warmup
        """
        # Alphas should already be [batch_size, num_classes] after forward pass
        
        # During warmup, train like regular classifier
        if epoch < self.warmup_epochs:
            # Use evidence (before adding 1) as logits for better numerical stability
            evidence = alphas - 1.0  # Convert back to evidence
            ce_loss = F.cross_entropy(evidence, targets, weight=self.class_weights)
            return ce_loss, ce_loss, torch.tensor(0.0), torch.tensor(0.0)
        
        # After warmup, use evidential formulation
        # Sum of alphas (strength of Dirichlet)
        alpha_sum = torch.sum(alphas, dim=1, keepdim=True)
        
        # Expected probabilities
        prob = alphas / alpha_sum
        
        # 1. Cross-entropy loss using expected probabilities with class weighting
        ce_loss = F.cross_entropy(torch.log(prob + 1e-8), targets, weight=self.class_weights)
        
        # 2. Evidence regularization - encourage confident predictions for correct class
        evidence = alphas - 1.0  # Convert back to evidence
        # Create one-hot encoding for targets
        targets_one_hot = F.one_hot(targets, num_classes=self.num_classes).float()
        
        # Regularize evidence: encourage high evidence for correct class, low for incorrect
        correct_evidence = torch.sum(evidence * targets_one_hot, dim=1)
        incorrect_evidence = torch.sum(evidence * (1 - targets_one_hot), dim=1)
        
        # Penalty: encourage evidence > 1 for correct class, evidence < 1 for incorrect classes
        evidence_reg = torch.mean(F.relu(1.0 - correct_evidence) + F.relu(incorrect_evidence - 1.0))
        
        # 3. Progressive annealing
        warmup_progress = (epoch - self.warmup_epochs) / self.annealing_epochs
        annealing_factor = min(1.0, max(0.0, warmup_progress))
        
        # Total loss
        total_loss = ce_loss + annealing_factor * self.annealing_coeff * evidence_reg
        
        return total_loss, ce_loss, evidence_reg, torch.tensor(0.0)


def train_evidential_classifier(model, dataloader, optimizer, loss_fn, device, epoch):
    """
    Training function for Evidential Deep Learning
    """
    model.train()
    total_loss = 0
    total_nll = 0
    total_kl = 0
    total_var_reg = 0
    correct = 0
    total = 0
    
    for data in tqdm.tqdm(dataloader):
        optimizer.zero_grad()

        X = data[0].to(device, dtype=torch.float)
        labels = data[1].to(device, dtype=torch.int).long()
        X = X.view(X.size(0), 1, X.size(1))
        
        alphas, evidence, _ = model(X)
        
        loss, nll, kl_div, var_reg = loss_fn(alphas, labels, epoch)
        loss.backward()
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        total_loss += loss.item()
        total_nll += nll.item()
        total_kl += kl_div.item()
        total_var_reg += var_reg.item()
        
        # Calculate accuracy
        alpha_sum = torch.sum(alphas, dim=1)
        pred_probs = alphas / alpha_sum.unsqueeze(1)
        predicted = torch.argmax(pred_probs, dim=1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    n_samples = len(dataloader.dataset)
    return (total_loss / n_samples, 
            total_nll / n_samples, 
            total_kl / n_samples,
            total_var_reg / n_samples,
            100.0 * correct / total)


def evaluate_evidential_model(model, test_loader, device):
    """
    Evaluate Evidential model with uncertainty quantification
    """
    model.eval()
    
    all_predictions = []
    all_epistemic = []
    all_aleatoric = []
    all_total_unc = []
    all_confidences = []
    all_alpha_sums = []
    all_targets = []
    
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data in test_loader:
            X = data[0].to(device, dtype=torch.float)
            labels = data[1].to(device, dtype=torch.int).long()
            X = X.view(X.size(0), 1, X.size(1))
            
            prob, epistemic, aleatoric, total_unc, confidence, alpha_sum = model.predict_with_uncertainty(X)
            
            all_predictions.append(prob.cpu())
            all_epistemic.append(epistemic.cpu())
            all_aleatoric.append(aleatoric.cpu())
            all_total_unc.append(total_unc.cpu())
            all_confidences.append(confidence.cpu())
            all_alpha_sums.append(alpha_sum.cpu())
            all_targets.append(labels.cpu())
            
            # Calculate accuracy
            predicted = torch.argmax(prob, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    predictions = torch.cat(all_predictions, dim=0)  # Concatenate along batch dimension
    epistemic_unc = torch.cat(all_epistemic, dim=0)
    aleatoric_unc = torch.cat(all_aleatoric, dim=0)
    total_uncertainty = torch.cat(all_total_unc, dim=0)
    confidences = torch.cat(all_confidences, dim=0)
    alpha_sums = torch.cat(all_alpha_sums, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    accuracy = 100.0 * correct / total
    
    return (predictions, epistemic_unc, aleatoric_unc, total_uncertainty, 
            confidences, alpha_sums, targets, accuracy)


if __name__ == '__main__':
    X_path = ['data/X_train_860.npy']
    y_path = ['data/y_train.npy']

    epochs = 150
    model_name = 'evidential_model'
    batch_size = 32
    learning_rate = 0.002  # Higher learning rate
    num_classes = 2
    annealing_coeff = 0.01  # Lower regularization initially
    regularization_coeff = 0.001
    
    dataset = ImpactEchoDatasetClassifier(X_path, y_path=y_path, array_size=860)
    print(f"Total number of training samples: {len(dataset)}")
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Calculate class weights for imbalanced data
    import numpy as np
    y_data = np.load(y_path[0])
    y_data[y_data < 1] = 0
    y_data[y_data > 0] = 1
    class_counts = np.bincount(y_data.astype(int))
    total_samples = len(y_data)
    class_weights = torch.FloatTensor([total_samples / (2 * count) for count in class_counts]).to(device)
    print(f"Class distribution: {class_counts}")
    print(f"Class weights: {class_weights}")

    # Initialize Evidential model
    model = EvidentialIENet(num_classes=num_classes, verbose=False).to(device)

    # Initialize Evidential loss with warmup and class weighting
    warmup_epochs = 30
    annealing_epochs = max(50, epochs - warmup_epochs)  # Use remaining epochs or minimum 50
    evidential_loss = EvidentialLoss(
        num_classes=num_classes,
        annealing_coeff=annealing_coeff,
        regularization_coeff=regularization_coeff,
        warmup_epochs=warmup_epochs,
        annealing_epochs=annealing_epochs,
        class_weights=class_weights
    )
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=10, factor=0.5)

    print('Training Evidential Deep Learning classifier...')
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Annealing coefficient: {annealing_coeff}")
    print(f"Regularization coefficient: {regularization_coeff}")

    best_loss = float('inf')
    for epoch in range(epochs):
        loss, nll, kl_div, var_reg, train_acc = train_evidential_classifier(
            model, dataloader, optimizer, evidential_loss, device, epoch
        )
        
        current_lr = optimizer.param_groups[0]['lr']
        phase = "Warmup" if epoch < warmup_epochs else "Evidential"
        print(f'Epoch {epoch:03d} ({phase}), Loss: {loss:.4f}, CE: {nll:.4f}, Reg: {kl_div:.4f}, '
              f'Acc: {train_acc:.2f}%, LR: {current_lr:.6f}')
        
        # Log to tensorboard
        writer.add_scalar("Loss/total", loss, epoch)
        writer.add_scalar("Loss/nll", nll, epoch)
        writer.add_scalar("Loss/kl_divergence", kl_div, epoch)
        writer.add_scalar("Loss/variance_reg", var_reg, epoch)
        writer.add_scalar("Accuracy/train", train_acc, epoch)
        writer.add_scalar("Learning_Rate", optimizer.param_groups[0]['lr'], epoch)
        
        if loss < best_loss:
            print(f"Best Loss: {loss:.4f}")
            best_loss = loss
            save_model(model, f'weights/{model_name}.pth')
            
        scheduler.step(loss)  # ReduceLROnPlateau needs loss value

    # Test Evidential uncertainty estimation
    print("\nEvaluating Evidential uncertainty estimation...")
    
    # Load test data
    X_test_path = ['data/X_test_860.npy']
    y_test_path = ['data/y_test.npy']
    test_dataset = ImpactEchoDatasetClassifier(X_test_path, y_path=y_test_path, array_size=860)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    (predictions, epistemic_unc, aleatoric_unc, total_unc, 
     confidences, alpha_sums, targets, test_accuracy) = evaluate_evidential_model(model, test_loader, device)
    
    print(f"Test Accuracy: {test_accuracy:.2f}%")
    
    # Uncertainty statistics
    print(f"Mean Epistemic Uncertainty: {epistemic_unc.mean():.4f} ± {epistemic_unc.std():.4f}")
    print(f"Mean Aleatoric Uncertainty: {aleatoric_unc.mean():.4f} ± {aleatoric_unc.std():.4f}")
    print(f"Mean Total Uncertainty: {total_unc.mean():.4f} ± {total_unc.std():.4f}")
    print(f"Mean Confidence: {confidences.mean():.4f} ± {confidences.std():.4f}")
    print(f"Mean Alpha Sum (Evidence Strength): {alpha_sums.mean():.4f} ± {alpha_sums.std():.4f}")
    
    # High uncertainty samples
    high_unc_threshold = total_unc.mean() + 2 * total_unc.std()
    high_unc_samples = (total_unc > high_unc_threshold).sum()
    print(f"High Uncertainty Samples (>μ+2σ): {high_unc_samples}")
    
    # Save results
    torch.save({
        'predictions': predictions,
        'epistemic_uncertainty': epistemic_unc,
        'aleatoric_uncertainty': aleatoric_unc,
        'total_uncertainty': total_unc,
        'confidences': confidences,
        'alpha_sums': alpha_sums,
        'targets': targets,
        'test_accuracy': test_accuracy,
        'model_config': {
            'num_classes': num_classes,
            'annealing_coeff': annealing_coeff,
            'regularization_coeff': regularization_coeff,
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate
        }
    }, f'weights/{model_name}_results.pth')
    
    print(f"Evidential results saved to weights/{model_name}_results.pth")

    writer.flush()
    writer.close()