# Imports
import torch
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt


# ----------------------------------------------------------------------------------------------------------------------

# ----------------------------------------------------------------------------------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------
class CAMERA_CNN_NET(nn.Module):
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(13, 10),
      nn.ReLU(),
      nn.Linear(10, 1)
    )

  def forward(self, x):
    return self.layers(x)


class LIDAR_CNN_NET(nn.Module):
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(13, 10),
      nn.ReLU(),
      nn.Linear(10, 1)
    )

  def forward(self, x):
    return self.layers(x)


class RADAR_CNN_NET(nn.Module):
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(13, 10),
      nn.ReLU(),
      nn.Linear(10, 1)
    )

  def forward(self, x):
    return self.layers(x)


class CAMERA_FC_NET(nn.Module):
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(13, 10),
      nn.ReLU(),
      nn.Linear(10, 1)
    )

  def forward(self, x):
    return self.layers(x)


class LIDAR_FC_NET(nn.Module):
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(13, 10),
      nn.ReLU(),
      nn.Linear(10, 1)
    )

  def forward(self, x):
    return self.layers(x)


class RADAR_FC_NET(nn.Module):
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(13, 10),
      nn.ReLU(),
      nn.Linear(10, 1)
    )

  def forward(self, x):
    return self.layers(x)

class ROAD_CONDITIONS_FC_NET(nn.Module):
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(13, 10),
      nn.ReLU(),
      nn.Linear(10, 1)
    )

  def forward(self, x):
    return self.layers(x)

class OUTPUT_FC_NET(nn.Module):
  def __init__(self):
    super().__init__()
    self.layers = nn.Sequential(
      nn.Linear(13, 10),
      nn.ReLU(),
      nn.Linear(10, 1)
    )

  def forward(self, x):
    return self.layers(x)
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.Conv_1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1)
        self.Conv_2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.Conv_3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.Conv_4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1)
        self.Conv_5 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.Conv_6 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1)

        self.Maxpool_2d = nn.MaxPool2d(2, 2)

        self.Linear_1 = nn.Linear(32 * 128, 1024)
        self.Linear_2 = nn.Linear(1024, 512)
        self.Linear_3 = nn.Linear(512, 10)

        # Random initialization
        nn.init.xavier_uniform_(self.Linear_1.weight)
        nn.init.xavier_uniform_(self.Linear_2.weight)
        nn.init.xavier_uniform_(self.Linear_3.weight)

    def forward(self, x):
        x = F.relu(self.Conv_1(x))
        x = F.relu(self.Conv_2(x))
        x = self.Maxpool_2d(x)

        x = F.relu(self.Conv_3(x))
        x = F.relu(self.Conv_4(x))
        x = self.Maxpool_2d(x)

        x = self.Conv_5(x)
        x = F.relu(self.Conv_6(x))
        x = self.Maxpool_2d(x)

        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.Linear_1(x))
        x = F.relu(self.Linear_2(x))
        x = self.Linear_3(x)
        return x