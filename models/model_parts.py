import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, verbose = False):
        super().__init__()

        self.verbose = verbose
        self.dropout = nn.Dropout(0.2)
        self.conv1 = nn.Conv1d(in_channels, 
                               out_channels, 
                               kernel_size=kernel_size, 
                               stride=stride, 
                               padding='same')
        self.bn_1 = nn.BatchNorm1d(out_channels)
        self.conv_2 = nn.Conv1d(out_channels, 
                                out_channels, 
                                kernel_size=kernel_size, 
                                stride=stride, 
                                padding='same')
        self.bn_2 = nn.BatchNorm1d(out_channels)
        self.conv_x = nn.Conv1d(in_channels, 
                                out_channels, 
                                kernel_size=kernel_size, 
                                stride=stride, 
                                padding='same')
        self.maxpooling1d_1 = nn.MaxPool1d(2, stride=2)

    def forward(self, x):

        if self.verbose:
            print(x.shape)
        xx = self.conv1(x)
        if self.verbose:
            print(xx.shape)
        xx = self.bn_1(xx)
        if self.verbose:
            print(xx.shape)
        xx = F.relu(xx)

        if self.verbose:
            print(xx.shape)
        xx = self.conv_2(xx)
        if self.verbose:
            print(xx.shape)
        bn_2 = self.bn_2(xx)
        bn_2 = self.dropout(bn_2)

        convx = self.conv_x(x)
        if self.verbose:
            print(convx.shape)
        out = bn_2 + convx

        if self.verbose:
            print(out.shape)
        out = F.relu(out)

        if self.verbose:
            print(out.shape)
        out = self.maxpooling1d_1(out)

        if self.verbose:
            print(out.shape)
        return out

# from torchinfo import summary
# model = ResidualBlock(1, 8, 200)
# summary(model, input_size=(1, 1, 860))