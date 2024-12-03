import torch
from PIL import Image
from torchvision import transforms
from mobileNetv3 import MobileNetV3WithEmbedding
from scipy.spatial.distance import cosine

# Hàm trích xuất embedding từ ảnh
def get_embedding(image_path, model, device):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        embedding = model(image)  # Trích xuất embedding
    
    return embedding.cpu().numpy()

# Hàm tính cosine similarity giữa các embedding
def compare_embeddings(embedding1, embedding2):
    embedding1 = embedding1.flatten()
    embedding2 = embedding2.flatten()
    similarity = 1 - cosine(embedding1, embedding2)
    return similarity