import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MLP
from models.model_parts import ResidualBlock


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
        self.bilstm_1 = nn.LSTM(input_size=832, hidden_size=128, num_layers=1, batch_first=True, bidirectional=True)

        # Encoder head
        self.linear_1 = nn.Linear(in_features=256, out_features=256)
        
        # Projection head
        self.mlp = MLP([256, 512, 128], norm=None)

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

        # reshape x to be 1d vector
        x = x.view(x.size(0), -1)
        if self.verbose:
            print(f"reshape_1: {x.shape}")

        x = nn.Flatten()(x)
        x = x.unsqueeze(0)
        if self.verbose:
            print(f"flatten_1: {x.shape}")

        x, _ = self.bilstm_1(x)
        if self.verbose:
            print(f"bilstm_1: {x.shape}")
        
        encoder = self.linear_1(x)
        if self.verbose:
            print(f"linear_1: {encoder.shape}")

        if train:
            projection = self.mlp(encoder)
            if self.verbose:
                print(f"mlp: {projection.shape}")
            return encoder, projection
        
        return encoder
    
# from torchinfo import summary
# model = EchoNet(verbose=True)
# summary(model, input_size=(1, 1, 860))
    
class Classifier(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.classifier_layer_1 = nn.Linear(in_features=256, out_features=2)

    def forward(self, x):
        return self.classifier_layer_1(x)
