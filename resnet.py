import torch.nn as nn
import torchvision.models as models


class Resnet18(nn.Module):
    def __init__(self):
        super().__init__()
        # Use weights arg to avoid deprecation warnings
        net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)

        # Expose layers with names that match common checkpoints (conv1/bn1/relu/maxpool)
        self.conv1 = net.conv1
        self.bn1 = net.bn1
        self.relu = net.relu
        self.maxpool = net.maxpool

        self.layer1 = net.layer1   # /4, 64ch
        self.layer2 = net.layer2   # /8, 128ch
        self.layer3 = net.layer3   # /16, 256ch
        self.layer4 = net.layer4   # /32, 512ch

    def forward(self, x):
        x = self.conv1(x)          # /2
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)        # /4

        x1 = self.layer1(x)        # /4, 64
        x2 = self.layer2(x1)       # /8, 128
        x3 = self.layer3(x2)       # /16, 256
        x4 = self.layer4(x3)       # /32, 512

        # Return features at /8, /16, /32 as expected by ContextPath
        return x2, x3, x4