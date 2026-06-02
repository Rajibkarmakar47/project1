import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    """A standard engineering building block for robust deep learning feature extraction"""
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x) # The residual skip connection prevents vanishing gradients
        out = F.relu(out)
        return out

class RobotGestureResNet(nn.Module):
    """Spatio-Temporal Residual network designed for Edge Hardware"""
    def __init__(self, num_classes):
        super(RobotGestureResNet, self).__init__()
        # Input channel is 1 (treating our grayscale spatio-temporal map like a 1-channel image)
        self.init_conv = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(32)
        
        # Stacking feature map processing layers
        self.layer1 = ResidualBlock(32, 64, stride=1)
        self.layer2 = ResidualBlock(64, 128, stride=2) # Dimensionality reduction via stride
        
        # Global pooling collapses spatial/temporal axes down cleanly
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(128, num_classes)

    def forward(self, x):
        # Input shape: (Batch_Size, 1, 30, 34)
        out = F.relu(self.bn(self.init_conv(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.global_pool(out)
        out = out.view(out.size(0), -1) # Flatten map features
        out = self.linear(out)
        return out