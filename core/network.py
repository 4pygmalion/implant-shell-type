import torch
import torch.nn as nn
import torch.nn.functional as F


def add_guassian_noise(x, stddev: float = 0.02) -> torch.Tensor:
    return x + torch.randn_like(x) * stddev


# Define the custom model class
class CustomModel(nn.Module):
    def __init__(self, noise: float = 0.02):
        super(CustomModel, self).__init__()

        self.noise = noise
        
        # self.conv1 = nn.Conv2d(3, 32, kernel_size=(8, 8), stride=2, padding=0)
        # self.conv2 = nn.Conv2d(32, 32, kernel_size=(8, 8), stride=2, padding=0)
        # self.conv3 = nn.Conv2d(32, 64, kernel_size=(8, 8), stride=2, padding=0)
        # self.maxpool = nn.AdaptiveMaxPool2d(output_size=(4, 4))       
        # self.maxpool = nn.MaxPool2d(
            # kernel_size=(2, 2), stride=2, ceil_mode=True, padding=0
        # ) 
        # self.fc1 = nn.Linear(1024, 250)
        
        # Define the layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(3, 3), padding=0)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding=0)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=(3, 3), padding=0)
        self.maxpool = nn.MaxPool2d(
            kernel_size=(2, 2), stride=2, ceil_mode=True, dilation=3, padding=1
        )
       
        self.fc1 = nn.Linear(256, 250)
        self.fc2 = nn.Linear(250, 100)
        self.fc3 = nn.Linear(100, 1)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv2(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.conv3(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = torch.flatten(x, start_dim=1)

        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        if self.training:
            x = add_guassian_noise(x)

        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        if self.training:
            x = add_guassian_noise(x)

        x = self.fc3(x)

        return x.squeeze()
