import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import math

class h_swish(nn.Module):
    def forward(self, x):
        return x * F.relu6(x + 3, inplace=True) / 6

class MobileNetV3WithEmbedding(nn.Module):
    def __init__(self, embedding_dim=128, output_channel=1024):
        super(MobileNetV3WithEmbedding, self).__init__()
        self.mobilenetv3 = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)

        # self.classifier = nn.Sequential(
        #     nn.Linear(self.mobilenetv3.classifier[3].in_features, output_channel),  # Sử dụng đúng số lượng đặc trưng đầu ra
        #     h_swish(),
        #     nn.BatchNorm1d(output_channel),
        #     nn.Dropout(0.2),
        #     nn.Linear(output_channel, embedding_dim)
        # )
        self.classifier = nn.Sequential(
            nn.Linear(576, embedding_dim),  # Chuyển từ 576 đầu vào xuống embedding_dim (128)
        )
    
    def forward(self, x):
        # Pass qua các lớp của MobileNetV3
        x = self.mobilenetv3.features(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        print("Feature extractor output size:", x.shape) 

        # Pass qua lớp classifier
        x = self.classifier(x)
        print("Classifier output size:", x.shape) 
        return x

