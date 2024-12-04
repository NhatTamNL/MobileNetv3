import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from mobileNetv3 import MobileNetV3WithEmbedding, MobileNetV3_FaceRecognition
from triplet_loss import TripletLoss
from data_triplets import FaceDataset


# Thiết lập các tham số
embedding_dim = 128
batch_size = 64
learning_rate = 0.001
epochs = 50
dataset = "ExtractedFaces"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tạo mô hình
# model = MobileNetV3WithEmbedding(embedding_dim=128).to(device)
model = MobileNetV3_FaceRecognition(embedding_dim=128).to(device)


optimizer = optim.Adam(model.parameters(), lr=learning_rate)
loss_fn = TripletLoss(margin=1.0)


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

print("Dataset Loading.........\n ")
dataset = FaceDataset(dataset, transform=transform)

print("Dataset Done.........\n ")

dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
print("DataLoader Loading.........\n ")

for epoch in range(epochs):
    model.train()  # Đặt chế độ train
    epoch_loss = 0.0

    for batch in dataloader:
        # Lấy batch dữ liệu
        anchor, positive, negative = batch
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

        # # Forward pass
        anchor_output = model(anchor)
        positive_output = model(positive)
        negative_output = model(negative)

        # Tính loss
        loss = loss_fn(anchor_output, positive_output, negative_output)

        # Backward pass và cập nhật trọng số
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Cập nhật tổng loss cho epoch
        epoch_loss += loss.item()

    # Hiển thị loss trung bình mỗi epoch
    print(f"Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss/len(dataloader):.4f}")
    if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), f"models_test/mobilenetv3_epoch{epoch+1}.pth")


# # Đánh giá mô hình
# model.eval()  # Đặt chế độ đánh giá
# with torch.no_grad():
#     for batch in dataloader:
#         anchor, positive, negative = batch
        
#         # Tính khoảng cách giữa anchor và positive
#         distance = torch.nn.functional.pairwise_distance(anchor, positive)
#         print("Distance between anchor and positive:", distance)