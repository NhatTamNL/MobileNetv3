import os
import torch
import numpy as np
from utils import get_embedding
from mobileNetv3 import MobileNetV3WithEmbedding

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MobileNetV3WithEmbedding(embedding_dim=128).to(device)
model.load_state_dict(torch.load('models/mobilenetv3_epoch50.pth'))
model.eval()

# Dataset Facebank
facebank_dir = "facebank/"  # Thư mục chứa các ảnh trong Facebank
facebank = {}

# Tạo embedding cho từng người trong Facebank
for person_name in os.listdir(facebank_dir):
    person_dir = os.path.join(facebank_dir, person_name)
    embeddings = []
    
    for img_name in os.listdir(person_dir):
        img_path = os.path.join(person_dir, img_name)
        print(img_path,"img_path\n")
        embedding = get_embedding(img_path, model, device)
        print(embedding,"embedding\n")
        embeddings.append(embedding)
        # embeddings_tensor = [torch.tensor(embedding) for embedding in embeddings]

    # Lấy trung bình các embedding cho mỗi người
    # facebank[person_name] = torch.mean(torch.stack(embeddings_tensor), dim=0)
    embeddings_array = np.stack([np.array(embedding) for embedding in embeddings])
    facebank[person_name] = np.mean(embeddings_array, axis=0)

# Lưu Facebank
# torch.save(facebank, "facebank.pth")
np.save("facebank_binary.npy", facebank)
print("Facebank saved successfully!")
