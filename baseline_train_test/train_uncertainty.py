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
from utils import *

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()

# configure logger
import logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

from models.model_parts import ResidualBlock


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

        # Reshape and flatten
        x = x.view(x.size(0), -1)
        if self.verbose:
            print(f"reshape_1: {x.shape}")

        x = nn.Flatten()(x)
        x = x.unsqueeze(0)
        if self.verbose:
            print(f"flatten_1: {x.shape}")

        # LSTM layers with dropout
        x, _ = self.bilstm_1(x)
        x = self.dropout1(x)
        if self.verbose:
            print(f"bilstm_1: {x.shape}")

        x, _ = self.bilstm_2(x)
        x = self.dropout2(x)
        if self.verbose:
            print(f"bilstm_2: {x.shape}")

        features, _ = self.bilstm_3(x)
        if self.verbose:
            print(f"bilstm_3: {features.shape}")
        
        # Output predictions
        mean_logits = self.classifier_mean(features)
        log_var = self.classifier_logvar(features)
        
        if self.verbose:
            print(f"mean_logits: {mean_logits.shape}")
            print(f"log_var: {log_var.shape}")
        
        return mean_logits, log_var, features

    def predict_with_uncertainty(self, x, mc_samples=50):
        """
        Perform Monte Carlo sampling to estimate epistemic uncertainty
        """
        predictions = []
        aleatoric_vars = []
        
        for _ in range(mc_samples):
            self.train()  # Enable dropout for this sample
            with torch.no_grad():
                mean_logits, log_var, _ = self.forward(x)
                predictions.append(F.softmax(mean_logits, dim=-1))
                aleatoric_vars.append(torch.exp(log_var))
        
        # Stack predictions
        predictions = torch.stack(predictions)  # [mc_samples, seq_len, batch_size, num_classes]
        aleatoric_vars = torch.stack(aleatoric_vars)
        
        # Calculate epistemic uncertainty (variance across MC samples)
        mean_prediction = torch.mean(predictions, dim=0)
        epistemic_uncertainty = torch.var(predictions, dim=0)
        
        # Calculate aleatoric uncertainty (average of predicted variances)
        aleatoric_uncertainty = torch.mean(aleatoric_vars, dim=0)
        
        # Total uncertainty
        total_uncertainty = epistemic_uncertainty + aleatoric_uncertainty
        
        return mean_prediction, epistemic_uncertainty, aleatoric_uncertainty, total_uncertainty


class UncertaintyLoss(nn.Module):
    """
    Simplified uncertainty loss:
    1. Standard classification loss
    2. KL divergence regularization for uncertainty head
    """
    
    def __init__(self, beta=0.01):
        super(UncertaintyLoss, self).__init__()
        self.beta = beta  # Weight for uncertainty regularization
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, mean_logits, log_var, targets):
        # Extract logits from sequence - handle the LSTM output format
        # mean_logits and log_var shape: [1, batch_size, num_classes]
        mean_logits = mean_logits.squeeze(0)  # [batch_size, num_classes]
        log_var = log_var.squeeze(0)  # [batch_size, num_classes]
        
        # Standard classification loss
        cls_loss = self.ce_loss(mean_logits, targets)
        
        # Regularization: penalize very high or very low uncertainties
        # Encourage log_var to be around 0 (var around 1)
        uncertainty_reg = self.beta * torch.mean(torch.abs(log_var))
        
        total_loss = cls_loss + uncertainty_reg
        
        return total_loss, cls_loss, uncertainty_reg


def train_uncertainty_classifier():
    total_loss = 0
    total_cls_loss = 0
    total_unc_reg = 0
    
    for data in tqdm.tqdm(dataloader):
        model.train()
        optimizer.zero_grad()

        X = data[0].to(device, dtype=torch.float)
        labels = data[1].to(device, dtype=torch.int).long()
        X = X.view(X.size(0), 1, X.size(1))

        mean_logits, log_var, _ = model(X)
        
        loss, cls_loss, unc_reg = uncertainty_loss(mean_logits, log_var, labels)
        loss.backward()
        
        total_loss += loss.item()
        total_cls_loss += cls_loss.item()
        total_unc_reg += unc_reg.item()

        optimizer.step()

    return (total_loss / len(dataset), 
            total_cls_loss / len(dataset), 
            total_unc_reg / len(dataset))


def evaluate_uncertainty(model, test_loader, device, mc_samples=50):
    """
    Evaluate model with uncertainty quantification
    """
    model.eval()
    
    all_predictions = []
    all_epistemic = []
    all_aleatoric = []
    all_total_unc = []
    all_targets = []
    
    with torch.no_grad():
        for data in test_loader:
            X = data[0].to(device, dtype=torch.float)
            labels = data[1].to(device, dtype=torch.int).long()
            X = X.view(X.size(0), 1, X.size(1))
            
            pred, epistemic, aleatoric, total_unc = model.predict_with_uncertainty(X, mc_samples)
            
            all_predictions.append(pred.cpu())
            all_epistemic.append(epistemic.cpu())
            all_aleatoric.append(aleatoric.cpu())
            all_total_unc.append(total_unc.cpu())
            all_targets.append(labels.cpu())
    
    predictions = torch.cat(all_predictions, dim=0)
    epistemic_unc = torch.cat(all_epistemic, dim=0)
    aleatoric_unc = torch.cat(all_aleatoric, dim=0)
    total_uncertainty = torch.cat(all_total_unc, dim=0)
    targets = torch.cat(all_targets, dim=0)
    
    return predictions, epistemic_unc, aleatoric_unc, total_uncertainty, targets


if __name__ == '__main__':
    X_path = ['data/X_train_860.npy']
    y_path = ['data/y_train.npy']

    epochs = 100
    model_name = 'uncertainty_model'
    batch_size = 32
    dropout_rate = 0.2  # Reduce dropout to help learning
    learning_rate = 0.001  # Increase learning rate
    
    dataset = ImpactEchoDatasetClassifier(X_path, y_path=y_path, array_size=860)
    print(f"Total number of training samples: {len(dataset)}")
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize uncertainty model
    model = UncertaintyIENet(dropout_rate=dropout_rate, verbose=False).to(device)

    # Initialize uncertainty loss
    uncertainty_loss = UncertaintyLoss(beta=0.1)  # Increase uncertainty learning weight
    
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    print('Training uncertainty classifier...')
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    best_loss = float('inf')
    for epoch in range(epochs):
        loss, cls_loss, unc_reg = train_uncertainty_classifier()
        
        print(f'Epoch {epoch:03d}, Total Loss: {loss:.4f}, Cls Loss: {cls_loss:.4f}, Unc Reg: {unc_reg:.4f}')
        
        # Log to tensorboard
        writer.add_scalar("Loss/total", loss, epoch)
        writer.add_scalar("Loss/classification", cls_loss, epoch)
        writer.add_scalar("Loss/uncertainty_reg", unc_reg, epoch)
        
        if loss < best_loss:
            print(f"Best Loss: {loss:.4f}")
            best_loss = loss
            save_model(model, f'weights/{model_name}.pth')
            
        scheduler.step()

    # Test uncertainty estimation
    print("\nEvaluating uncertainty estimation...")
    
    # Load test data
    X_test_path = ['data/X_test_860.npy']
    y_test_path = ['data/y_test.npy']
    test_dataset = ImpactEchoDatasetClassifier(X_test_path, y_path=y_test_path, array_size=860)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    predictions, epistemic_unc, aleatoric_unc, total_unc, targets = evaluate_uncertainty(
        model, test_loader, device, mc_samples=50
    )
    
    # Calculate accuracy
    pred_classes = torch.argmax(predictions.squeeze(1), dim=-1)
    accuracy = (pred_classes == targets).float().mean()
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Uncertainty statistics
    print(f"Mean Epistemic Uncertainty: {epistemic_unc.mean():.4f} ± {epistemic_unc.std():.4f}")
    print(f"Mean Aleatoric Uncertainty: {aleatoric_unc.mean():.4f} ± {aleatoric_unc.std():.4f}")
    print(f"Mean Total Uncertainty: {total_unc.mean():.4f} ± {total_unc.std():.4f}")
    
    # Save uncertainty results
    torch.save({
        'predictions': predictions,
        'epistemic_uncertainty': epistemic_unc,
        'aleatoric_uncertainty': aleatoric_unc,
        'total_uncertainty': total_unc,
        'targets': targets,
        'accuracy': accuracy
    }, f'weights/{model_name}_uncertainty_results.pth')
    
    print(f"Uncertainty results saved to weights/{model_name}_uncertainty_results.pth")

    writer.flush()
    writer.close()