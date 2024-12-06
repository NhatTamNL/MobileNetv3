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

        self.classifier = nn.Sequential(
            nn.Linear(576, embedding_dim)  # Chuyển từ 576 đầu vào xuống embedding_dim (128)
        )
    
    def forward(self, x):
        # Pass qua các lớp của MobileNetV3
        x = self.mobilenetv3.features(x)
        x = F.adaptive_avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        # print("Feature extractor output size:", x.shape) 

        # Pass qua lớp classifier
        x = self.classifier(x)
        # print("Classifier output size:", x.shape) 
        return x


class MobileNetV3_FaceRecognition(nn.Module):
    def __init__(self, embedding_dim=128, output_channel=1024):
        super(MobileNetV3_FaceRecognition, self).__init__()
        
        # Tải MobileNetV3 đã huấn luyện sẵn
        mobilenet_v3 = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        
        # Lấy các lớp trích xuất đặc trưng từ MobileNetV3
        self.features = mobilenet_v3.features
        
        # Global Average Pooling
        self.pooling = nn.AdaptiveAvgPool2d(1)  # Đưa về kích thước (1, 1)
        
        # Các lớp MLP để học đặc trưng sau khi trích xuất
        self.fc1 = nn.Linear(576, 512)  # 576 là chiều đầu ra từ MobileNetV3
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 128)  # Lớp MLP thứ hai
        self.fc3 = nn.Linear(128, 128)  # Lớp MLP thứ ba để trích xuất đặc trưng cuối
        self.dropout = nn.Dropout(0.5)  # Giúp giảm overfitting
        
        # Lớp classifier cho Face Recognition (nếu bạn làm phân loại theo người)
        # if num_classes:
        self.classifier = nn.Linear(128, embedding_dim)  # Phân loại theo số lượng người
        
    def forward(self, x):
        # Trích xuất đặc trưng từ MobileNetV3
        x = self.features(x)
        
        # Áp dụng Global Average Pooling
        x = self.pooling(x)
        
        # Flatten đặc trưng trích xuất từ CNN
        x = torch.flatten(x, 1)
        
        # Áp dụng MLP để giảm chiều và học các quan hệ phi tuyến
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        
        # Lớp phân loại (chỉ cần nếu bạn làm phân loại người)
        # if hasattr(self, 'classifier'):
        x = self.classifier(x)
        
        return x

if __name__ == '__main__':
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = MobileNetV3_FaceRecognition(embedding_dim=128).to(device)
    
    from torchsummary import summary
    summary(model, input_size=(3, 224, 224))  # Xem thông tin model
    # input_tensor = torch.randn(1, 3, 224, 224).to(device)
    # embedding = model(input_tensor)
    # print(embedding.shape) 
# Ví dụ sử dụng
