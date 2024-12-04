from torchvision import transforms
from torch.utils.data import DataLoader
from data_triplets import FaceDataset

# Transform để chuẩn hóa ảnh
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Khởi tạo dataset
dataset = FaceDataset(root_dir='ExtractedFaces', transform=transform)

# Kiểm tra dữ liệu
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

for anchor, positive, negative in dataloader:
    print("Anchor shape:", anchor.shape)
    print("Positive shape:", positive.shape)
    print("Negative shape:", negative.shape)
    break
