import torch
import torch.nn as nn
import torchvision.transforms as transformers
from pillow import Image
import cv2
import numpy as np

# Create CNN Model
class MangaOCR(nn.Module):
    def __int__(self, num_classes):
        super(MangaOCR, self).__init__()

        # Creating Feature Map
        self.features = nn.Sequential(

            # 32 Feature map
            nn.Conv2d(1,32, kernel_size=3, stride=1, padding=1),    # 2D Convolution
            nn.ReLU(inplace=True),                                  # Relu Model
            nn.MaxPool2d(kernel_size=2, stride=2),                  # Compression Max Mooling

            # 64 feature map
            nn.Conv2d(32,64, kernel_size=3, stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # 128 feature map
            nn.Conv2d(64,128, kernel_size=3, stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        # Average pool 128 Features into 7x7 Grids, Flatten into linear vector
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(128 * 7 * 7, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(1024, num_classes),   # Num Classes = Number of characters we are trying to recognize
        )

        # Data Flow
        def forward(self,x):
            x = self.features(x)
            x = self.avgpool(x)
            x = torch.flatten(x,1)
            x = self.classifier(x)
            return x

