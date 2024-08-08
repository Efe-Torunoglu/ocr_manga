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
        self.features = nn.Sequential(

            # Layer one of the CNN
            nn.Conv2d(1,32, kernel_size=3, stride=1, padding=1),    # 2D Convolution
            nn.ReLU(inplace=True),                                  # Relu Model
            nn.MaxPool2d(kernel_size=2, stride=2),                  # Compression Max Mooling

            # Layer two of the CNN
            nn.Conv2d(32,64, kernel_size=3, stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Layer three of the CNN
            nn.Conv2d(64,128, kernel_size=3, stride=1,padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),


        )