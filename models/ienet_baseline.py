import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MLP
from model_parts import ResidualBlock


class IENet(nn.Module):

    def __init__(self, verbose = False):
        super(IENet, self).__init__()
        self.verbose = verbose
        self.residual_1 = ResidualBlock(1, 8, 200)
        self.residual_2 = ResidualBlock(8, 16, 100)
        self.residual_3 = ResidualBlock(16, 16, 50)
        self.residual_4 = ResidualBlock(16, 32, 25)
        self.residual_5 = ResidualBlock(32, 64, 13)
        self.residual_6 = ResidualBlock(64, 64, 7)
        self.bilstm_1 = nn.LSTM(input_size=832, hidden_size=32, num_layers=1, batch_first=True, bidirectional=True)
        self.bilstm_2 = nn.LSTM(input_size=64, hidden_size=32, num_layers=1, batch_first=True, bidirectional=True)
        self.bilstm_3 = nn.LSTM(input_size=64, hidden_size=32, num_layers=1, batch_first=True, bidirectional=True)

        # Encoder head
        self.linear_1 = nn.Linear(in_features=64, out_features=2)

    def forward(self, x):
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

        x, _ = self.bilstm_2(x)
        if self.verbose:
            print(f"bilstm_2: {x.shape}")

        bilstm3_output, _ = self.bilstm_3(x)
        if self.verbose:
            print(f"bilstm_3: {x.shape}")
        
        encoder = self.linear_1(x)
        if self.verbose:
            print(f"linear_1: {encoder.shape}")
        
        return encoder, bilstm3_output

# from torchinfo import summary
# model = IENet()
# summary(model, input_size=(1, 1, 860))