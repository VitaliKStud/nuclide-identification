import torch.nn as nn
import torch.nn.functional as F
from config.loader import load_config


class CNN(nn.Module):
    """
    CNN for classification of nuclides. Using config-loader to load the size for all layers.
    """

    def __init__(self, num_classes):
        super(CNN, self).__init__()
        self.config_loader = load_config()
        self.conv1 = nn.Conv1d(
            self.config_loader["cnn"]["input_dim"],
            self.config_loader["cnn"]["first_layer"],
            kernel_size=2,
        )
        self.bn1 = nn.BatchNorm1d(self.config_loader["cnn"]["first_layer"])
        self.pool1 = nn.MaxPool1d(self.config_loader["cnn"]["max_pool"])

        self.conv2 = nn.Conv1d(
            self.config_loader["cnn"]["first_layer"],
            self.config_loader["cnn"]["second_layer"],
            kernel_size=2,
            groups=2
        )
        self.bn2 = nn.BatchNorm1d(self.config_loader["cnn"]["second_layer"])
        self.pool2 = nn.MaxPool1d(self.config_loader["cnn"]["max_pool"])

        self.conv3 = nn.Conv1d(
            self.config_loader["cnn"]["second_layer"],
            self.config_loader["cnn"]["third_layer"],
            kernel_size=2,
            groups=2
        )
        self.bn3 = nn.BatchNorm1d(self.config_loader["cnn"]["third_layer"])
        self.pool3 = nn.AdaptiveAvgPool1d(self.config_loader["cnn"]["adaptive_pool"])

        self.fc1 = nn.Linear(self.config_loader["cnn"]["third_layer"] * 1, self.config_loader["cnn"]["fourth_layer"])
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(self.config_loader["cnn"]["fourth_layer"], num_classes)

    def forward(self, x):
        x = self.pool1(F.leaky_relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.leaky_relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.leaky_relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        return x
