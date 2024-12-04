import torch
from utils import get_embedding, compare_embeddings
from mobileNetv3 import MobileNetV3WithEmbedding
from data_triplets import FaceDataset  
from torchvision import transforms
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MobileNetV3WithEmbedding(embedding_dim=128).to(device)
model.load_state_dict(torch.load('models/mobilenetv3_epoch50.pth'))
model.eval()  

# Load Facebank
facebank = np.load("facebank_binary.npy", allow_pickle=True).item()

# Ảnh đầu vào để nhận diện
input_image_path = 'data_test/Joey00036.jpg'
input_embedding = get_embedding(input_image_path, model, device)

# Kiểm tra với các embedding trong Facebank
threshold = 0.7  # Ngưỡng cosine similarity để quyết định có phải là cùng một người không
recognized_person = "Unknown"

for person_name, face_embedding in facebank.items():
    similarity = compare_embeddings(input_embedding, face_embedding)
    print(similarity,"similarity\n")
    if similarity > threshold:
        recognized_person = person_name
        break

print(f"Đây là ảnh của: {recognized_person}")