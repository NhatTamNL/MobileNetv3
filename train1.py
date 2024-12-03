import torch
import torchvision.models as models
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader

# Tạo dataset và loader cho dữ liệu tập trung
class FaceDataset(Dataset):
    def __init__(self, images, labels):
        self.images = images
        self.labels = labels

    def __getitem__(self, index):
        image = self.images[index]
        label = self.labels[index]

        return {'image': torch.tensor(image), 'label': torch.tensor(label)}

    def __len__(self):
        return len(self.images)

# Tạo mô hình MobileNetV3
model = models.mobilenet_v3(pretrained=True)

# Chuyển model sang mode dự đoán (model.eval())
model.eval()

# Khởi tạo device (GPU nếu có sẵn)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# Tạo dataset và loader cho dữ liệu tập trung
train_dataset = FaceDataset(train_images, train_labels)
test_dataset = FaceDataset(test_images, test_labels)

batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

# Tạo loss function và optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# Train mô hình
for epoch in range(10): # 10 epochs
    for i in range(len(train_loader)):
        images, labels = train_loader[i]
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

    print(f'Epoch {epoch+1}, Loss: {loss.item()}')
