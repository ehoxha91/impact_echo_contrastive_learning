import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(
            in_channels=in_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            padding='same',
            bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.LeakyReLU()
        self.conv2 = nn.Conv1d(
            in_channels=out_channels, 
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            padding='same',
            bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.convX = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels, 
            kernel_size=kernel_size, 
            padding='same',
            bias=False
        )
        self.relu2 = nn.LeakyReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)

        # Weight initialization
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.conv2.weight)
        nn.init.kaiming_normal_(self.convX.weight)

    def forward(self, x):
        conv1 = self.conv1(x)
        bn1 = self.bn1(conv1)
        relu1 = self.relu1(bn1)
        conv2 = self.conv2(relu1)
        bn2 = self.bn2(conv2)
        convX = self.convX(x)
        residual = bn2 + convX
        relu2 = self.relu2(residual)
        mpooling = self.maxpool(relu2)
        return mpooling
    
# from torchinfo import summary
# model = ResidualBlock(1, 8, 200)
# summary(model, input_size=(1, 1, 860))