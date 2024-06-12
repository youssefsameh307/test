import torch
import torch.nn as nn
import torch.nn.functional as F

class FCN(nn.Module):
    def __init__(self,output_classes):
        super(FCN, self).__init__()
        self.fc1 = nn.Linear(3 * 28 * 28, 128)  # Input size: 28x28, Output size: 128
        self.fc2 = nn.Linear(128, 64)       # Input size: 128, Output size: 64
        self.fc3 = nn.Linear(64, output_classes)         # Input size: 64, Output size: 10 (assuming 10 classes)

    def forward(self, x):
        # Flatten the input image tensor
        x = x.view(-1, 3 * 28 * 28)
        # Apply first fully connected layer with ReLU activation
        x = F.relu(self.fc1(x))
        # Apply second fully connected layer with ReLU activation
        x = F.relu(self.fc2(x))
        # Apply third fully connected layer (output layer)
        x = self.fc3(x)
        return x
