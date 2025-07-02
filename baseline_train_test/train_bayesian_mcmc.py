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
import math
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


class BayesianLinear(nn.Module):
    """
    Bayesian Linear Layer with learnable weight and bias distributions
    """
    def __init__(self, in_features, out_features, prior_var=1.0):
        super(BayesianLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.prior_var = prior_var
        
        # Weight mean and log variance parameters
        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_logvar = nn.Parameter(torch.Tensor(out_features, in_features))
        
        # Bias mean and log variance parameters
        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_logvar = nn.Parameter(torch.Tensor(out_features))
        
        # Initialize parameters
        self.reset_parameters()
    
    def reset_parameters(self):
        # Initialize weight parameters
        stdv = 1. / math.sqrt(self.weight_mu.size(1))
        self.weight_mu.data.uniform_(-stdv, stdv)
        self.weight_logvar.data.fill_(-5.0)  # Start with very low variance
        
        # Initialize bias parameters
        self.bias_mu.data.uniform_(-stdv, stdv)
        self.bias_logvar.data.fill_(-5.0)
    
    def forward(self, input, sample=True):
        if sample:
            # Sample weights and biases from distributions
            weight_std = torch.exp(0.5 * self.weight_logvar)
            bias_std = torch.exp(0.5 * self.bias_logvar)
            
            weight_eps = torch.randn_like(self.weight_mu)
            bias_eps = torch.randn_like(self.bias_mu)
            
            weight = self.weight_mu + weight_std * weight_eps
            bias = self.bias_mu + bias_std * bias_eps
        else:
            # Use mean values for deterministic inference
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(input, weight, bias)
    
    def kl_divergence(self):
        """
        Compute KL divergence between learned distributions and priors
        """
        # KL divergence for weights
        weight_var = torch.exp(self.weight_logvar)
        weight_kl = 0.5 * torch.sum(
            self.weight_mu.pow(2) / self.prior_var + 
            weight_var / self.prior_var - 
            self.weight_logvar + 
            math.log(self.prior_var) - 1
        )
        
        # KL divergence for biases
        bias_var = torch.exp(self.bias_logvar)
        bias_kl = 0.5 * torch.sum(
            self.bias_mu.pow(2) / self.prior_var + 
            bias_var / self.prior_var - 
            self.bias_logvar + 
            math.log(self.prior_var) - 1
        )
        
        return weight_kl + bias_kl


class BayesianIENet(nn.Module):
    """
    Bayesian IENet with uncertainty quantification through weight distributions
    """
    
    def __init__(self, prior_var=1.0, verbose=False):
        super(BayesianIENet, self).__init__()
        self.verbose = verbose
        self.prior_var = prior_var
        
        # Feature extraction layers (deterministic)
        self.residual_1 = ResidualBlock(1, 8, 200)
        self.residual_2 = ResidualBlock(8, 16, 100)
        self.residual_3 = ResidualBlock(16, 16, 50)
        self.residual_4 = ResidualBlock(16, 32, 25)
        self.residual_5 = ResidualBlock(32, 64, 13)
        self.residual_6 = ResidualBlock(64, 64, 7)
        
        # LSTM layers (deterministic)
        self.bilstm_1 = nn.LSTM(input_size=832, hidden_size=32, num_layers=1, 
                               batch_first=True, bidirectional=True)
        self.bilstm_2 = nn.LSTM(input_size=64, hidden_size=32, num_layers=1, 
                               batch_first=True, bidirectional=True)
        self.bilstm_3 = nn.LSTM(input_size=64, hidden_size=32, num_layers=1, 
                               batch_first=True, bidirectional=True)
        
        # Bayesian classification layer
        self.bayesian_classifier = BayesianLinear(64, 2, prior_var=prior_var)

    def forward(self, x, sample=True):
        # Feature extraction (deterministic)
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
        x = nn.Flatten()(x)
        x = x.unsqueeze(0)
        if self.verbose:
            print(f"flatten: {x.shape}")

        # LSTM layers (deterministic)
        x, _ = self.bilstm_1(x)
        if self.verbose:
            print(f"bilstm_1: {x.shape}")

        x, _ = self.bilstm_2(x)
        if self.verbose:
            print(f"bilstm_2: {x.shape}")

        features, _ = self.bilstm_3(x)
        if self.verbose:
            print(f"bilstm_3: {features.shape}")
        
        # Bayesian classification
        logits = self.bayesian_classifier(features, sample=sample)
        
        if self.verbose:
            print(f"logits: {logits.shape}")
        
        return logits, features

    def kl_divergence(self):
        """
        Total KL divergence for all Bayesian layers
        """
        return self.bayesian_classifier.kl_divergence()

    def predict_with_uncertainty(self, x, n_samples=100):
        """
        Perform Bayesian inference to estimate predictive uncertainty
        """
        self.eval()
        
        predictions = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                logits, _ = self.forward(x, sample=True)
                predictions.append(F.softmax(logits, dim=-1))
        
        # Stack predictions
        predictions = torch.stack(predictions)  # [n_samples, seq_len, batch_size, num_classes]
        
        # Calculate predictive mean and uncertainty
        mean_prediction = torch.mean(predictions, dim=0)
        predictive_uncertainty = torch.var(predictions, dim=0)
        
        # Calculate confidence (max probability)
        confidence = torch.max(mean_prediction, dim=-1)[0]
        
        return mean_prediction, predictive_uncertainty, confidence


class BayesianLoss(nn.Module):
    """
    Bayesian loss combining likelihood and KL divergence
    """
    
    def __init__(self, kl_weight=1.0, n_samples=3):
        super(BayesianLoss, self).__init__()
        self.kl_weight = kl_weight
        self.n_samples = n_samples
        self.ce_loss = nn.CrossEntropyLoss()
        
    def forward(self, model, x, targets, n_batches):
        # Sample multiple times and average the likelihood
        likelihood_loss = 0.0
        
        for _ in range(self.n_samples):
            logits, _ = model(x, sample=True)
            # Extract logits from sequence - handle the LSTM output format
            logits = logits.squeeze(0)  # [batch_size, num_classes]
            likelihood_loss += self.ce_loss(logits, targets)
        
        likelihood_loss /= self.n_samples
        
        # KL divergence (proper scaling for ELBO)
        kl_div = model.kl_divergence() / len(targets)  # Scale by batch size instead
        
        # Total loss
        total_loss = likelihood_loss + self.kl_weight * kl_div
        
        return total_loss, likelihood_loss, kl_div


def train_bayesian_classifier(model, dataloader, optimizer, loss_fn, n_batches, device):
    """
    Training function for Bayesian neural network
    """
    model.train()
    total_loss = 0
    total_likelihood = 0
    total_kl = 0
    
    for data in tqdm.tqdm(dataloader):
        optimizer.zero_grad()

        X = data[0].to(device, dtype=torch.float)
        labels = data[1].to(device, dtype=torch.int).long()
        X = X.view(X.size(0), 1, X.size(1))
        
        loss, likelihood, kl_div = loss_fn(model, X, labels, len(dataloader.dataset))
        loss.backward()
        
        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()

        total_loss += loss.item()
        total_likelihood += likelihood.item()
        total_kl += kl_div.item()

    n_samples = len(dataloader.dataset)
    return (total_loss / n_samples, 
            total_likelihood / n_samples, 
            total_kl / n_samples)


def evaluate_bayesian_model(model, test_loader, device, n_samples=100):
    """
    Evaluate Bayesian model with uncertainty quantification
    """
    model.eval()
    
    all_predictions = []
    all_uncertainties = []
    all_confidences = []
    all_targets = []
    
    with torch.no_grad():
        for data in test_loader:
            X = data[0].to(device, dtype=torch.float)
            labels = data[1].to(device, dtype=torch.int).long()
            X = X.view(X.size(0), 1, X.size(1))
            
            pred, uncertainty, confidence = model.predict_with_uncertainty(X, n_samples)
            
            all_predictions.append(pred.cpu())
            all_uncertainties.append(uncertainty.cpu())
            all_confidences.append(confidence.cpu())
            all_targets.append(labels.cpu())
    
    predictions = torch.cat(all_predictions, dim=1)  # Concatenate along batch dimension
    uncertainties = torch.cat(all_uncertainties, dim=1)
    confidences = torch.cat(all_confidences, dim=1)
    targets = torch.cat(all_targets, dim=0)
    
    return predictions, uncertainties, confidences, targets


if __name__ == '__main__':
    X_path = ['data/X_train_860.npy']
    y_path = ['data/y_train.npy']

    epochs = 150
    model_name = 'bayesian_mcmc_model'
    batch_size = 16  # Smaller batch size for Bayesian training
    learning_rate = 0.0005  # Lower learning rate for stable training
    prior_var = 1.0  # Prior variance for weights
    kl_weight = 0.001  # Lower KL divergence weight for better learning
    n_likelihood_samples = 5  # Number of samples for likelihood estimation
    
    dataset = ImpactEchoDatasetClassifier(X_path, y_path=y_path, array_size=860)
    print(f"Total number of training samples: {len(dataset)}")
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    n_batches = len(dataloader)

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize Bayesian model
    model = BayesianIENet(prior_var=prior_var, verbose=False).to(device)

    # Initialize Bayesian loss
    bayesian_loss = BayesianLoss(kl_weight=kl_weight, n_samples=n_likelihood_samples)
    
    # Use different learning rates for different parameters
    bayesian_params = []
    other_params = []
    
    for name, param in model.named_parameters():
        if 'bayesian' in name:
            bayesian_params.append(param)
        else:
            other_params.append(param)
    
    optimizer = torch.optim.Adam([
        {'params': other_params, 'lr': learning_rate},
        {'params': bayesian_params, 'lr': learning_rate * 2.0}  # Higher LR for Bayesian params
    ])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=25, gamma=0.8)

    print('Training Bayesian classifier with MCMC...')
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Prior variance: {prior_var}")
    print(f"KL weight: {kl_weight}")
    print(f"Likelihood samples per forward pass: {n_likelihood_samples}")

    best_loss = float('inf')
    for epoch in range(epochs):
        loss, likelihood, kl_div = train_bayesian_classifier(
            model, dataloader, optimizer, bayesian_loss, len(dataset), device
        )
        
        print(f'Epoch {epoch:03d}, Total Loss: {loss:.4f}, Likelihood: {likelihood:.4f}, KL Div: {kl_div:.4f}')
        
        # Log to tensorboard
        writer.add_scalar("Loss/total", loss, epoch)
        writer.add_scalar("Loss/likelihood", likelihood, epoch)
        writer.add_scalar("Loss/kl_divergence", kl_div, epoch)
        writer.add_scalar("Learning_Rate", optimizer.param_groups[0]['lr'], epoch)
        
        if loss < best_loss:
            print(f"Best Loss: {loss:.4f}")
            best_loss = loss
            save_model(model, f'weights/{model_name}.pth')
            
        scheduler.step()

    # Test Bayesian uncertainty estimation
    print("\nEvaluating Bayesian uncertainty estimation...")
    
    # Load test data
    X_test_path = ['data/X_test_860.npy']
    y_test_path = ['data/y_test.npy']
    test_dataset = ImpactEchoDatasetClassifier(X_test_path, y_path=y_test_path, array_size=860)
    test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    predictions, uncertainties, confidences, targets = evaluate_bayesian_model(
        model, test_loader, device, n_samples=100
    )
    
    # Calculate accuracy
    pred_classes = torch.argmax(predictions.squeeze(0), dim=-1)
    accuracy = (pred_classes == targets).float().mean()
    print(f"Test Accuracy: {accuracy:.4f}")
    
    # Uncertainty statistics
    mean_uncertainty = uncertainties.mean()
    mean_confidence = confidences.mean()
    
    print(f"Mean Predictive Uncertainty: {mean_uncertainty:.4f} ± {uncertainties.std():.4f}")
    print(f"Mean Confidence: {mean_confidence:.4f} ± {confidences.std():.4f}")
    
    # High uncertainty samples
    high_unc_threshold = mean_uncertainty + 2 * uncertainties.std()
    high_unc_samples = (uncertainties.squeeze(0)[:, 0] > high_unc_threshold).sum()
    print(f"High Uncertainty Samples (>μ+2σ): {high_unc_samples}")
    
    # Save results
    torch.save({
        'predictions': predictions,
        'uncertainties': uncertainties,
        'confidences': confidences,
        'targets': targets,
        'accuracy': accuracy,
        'model_config': {
            'prior_var': prior_var,
            'kl_weight': kl_weight,
            'n_likelihood_samples': n_likelihood_samples,
            'epochs': epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate
        }
    }, f'weights/{model_name}_results.pth')
    
    print(f"Bayesian results saved to weights/{model_name}_results.pth")

    writer.flush()
    writer.close()