import os
import torch
from utils import get_embedding, compare_embeddings, compare_correlation
from mobileNetv3 import MobileNetV3WithEmbedding, MobileNetV3_FaceRecognition
from data_triplets import FaceDataset  
from torchvision import transforms
import numpy as np

# Thiết bị sử dụng (GPU/CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Tải model
# model = MobileNetV3WithEmbedding(embedding_dim=128).to(device)
model = MobileNetV3_FaceRecognition(embedding_dim=128).to(device)

model.load_state_dict(torch.load('model_regv2/mobilenetv3_epoch100.pth'))
model.eval()  

# Load Facebank
facebank = np.load("facebank_binary.npy", allow_pickle=True).item()

# Thư mục chứa ảnh để nhận diện
# input_folder_path = 'data_test'  
input_folder_path = 'friend'
threshold = 0.9  # Ngưỡng cosine similarity để quyết định có phải là cùng một người không

# Duyệt qua các file trong thư mục
for file_name in os.listdir(input_folder_path):
    # Chỉ xử lý file ảnh (bỏ qua các file không phải ảnh)
    if not file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
        continue
    
    # Đường dẫn file ảnh
    input_image_path = os.path.join(input_folder_path, file_name)
    
    # Tính embedding của ảnh
    input_embedding = get_embedding(input_image_path, model, device)
    
    # So sánh với các embedding trong Facebank
    recognized_person = "Unknown"
    for person_name, face_embedding in facebank.items():
        similarity = compare_embeddings(input_embedding, face_embedding)
        print(f"{file_name} - Similarity with {person_name}: {similarity:.2f}")
        
        if similarity > threshold:
            recognized_person = person_name
            break
    
    print(f"Ảnh {file_name}: Đây là {recognized_person}")
