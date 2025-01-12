import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MLP
# aic_st_1
# from model_parts import ResidualBlock

# aic_st_2, for epoch 2 to 76
from models.model_parts2 import ResidualBlock


class EchoNet(nn.Module):

    def __init__(self, verbose = False):
        super(EchoNet, self).__init__()
        
        self.verbose = verbose
        self.residual_1 = ResidualBlock(1, 8, 200)
        self.residual_2 = ResidualBlock(8, 16, 100)
        self.residual_3 = ResidualBlock(16, 16, 50)
        self.residual_4 = ResidualBlock(16, 32, 25)
        self.residual_5 = ResidualBlock(32, 64, 13)
        self.residual_6 = ResidualBlock(64, 64, 7)
        encoder_layer = nn.TransformerEncoderLayer(d_model=64, nhead=32, dim_feedforward=512, dropout=0.1)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)
        self.projection_layer = MLP([832, 128, 64], norm=None)
        self.linear_1 = nn.Linear(in_features=64, out_features=2)

    def forward(self, x, train=True):
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

        x = x.permute(2, 0, 1)  # Transformer expects (seq_len, batch_size, feature_dim)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)  # Revert the permutation
        x = x.contiguous().view(x.size(0), -1)
        
        projection = self.projection_layer(x)
        projection = projection.view(1, projection.size(0), projection.size(1))
        if self.verbose:
            print(f"projection: {projection.shape}")
        return projection
    

class Classifier(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.classifier_layer_1 = nn.Linear(in_features=64, out_features=2)

    def forward(self, x):
        return self.classifier_layer_1(x)

from torchinfo import summary
model = EchoNet()
summary(model, input_size=(1, 1, 860))