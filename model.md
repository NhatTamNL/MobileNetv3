```
class MobileNetV3WithEmbedding(nn.Module):
    def __init__(self, embedding_dim=128):
        super(MobileNetV3WithEmbedding, self).__init__()
        # Load MobileNetV3 Small
        self.mobilenetv3 = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)

        # Trích xuất đặc trưng
        self.features = self.mobilenetv3.features

        # Global Average Pooling
        self.pooling = nn.AdaptiveAvgPool2d(1)

        # MLP nhẹ để học đặc trưng tốt hơn
        self.classifier = nn.Sequential(
            nn.Linear(576, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, embedding_dim)
        )

    def forward(self, x):
        # Trích xuất đặc trưng
        x = self.features(x)
        x = self.pooling(x)
        x = torch.flatten(x, 1)

        # Pass qua classifier
        x = self.classifier(x)
        return x

```