import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from mobileNetv3 import mobilenetv3_small, mobilenetv3_small_reduced


# 1. Cấu hình thiết bị
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on device: {device}")

# 2. Tạo dữ liệu và DataLoader


# 3. Tải dữ liệu từ thư mục 

# 4. Khỏi tạo mô hình 
model = mobilenetv3_small_reduced(num_classes=len(train_dataset.classes))  # Thay num_classes theo số lớp
model = model.to(device)

# 5. Loss function và Optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)


# 6. Vòng lặp huấn luyện
num_epochs = 10
best_acc = 0.0

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = 100. * correct / total
    print(f"Epoch [{epoch+1}/{num_epochs}] - Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.2f}%")

    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            val_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            val_total += labels.size(0)
            val_correct += predicted.eq(labels).sum().item()

    val_loss /= len(val_loader.dataset)
    val_acc = 100. * val_correct / val_total
    print(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")

    # Lưu mô hình nếu cải thiện
    if val_acc > best_acc:
        best_acc = val_acc
        torch.save(model.state_dict(), "best_mobilenetv3_small_reduced.pth")
        print("Model saved!")

print("Training complete!")