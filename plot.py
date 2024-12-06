import torch
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from torchvision import transforms
from torch.utils.data import DataLoader
from data_triplets import FaceDataset
from mobileNetv3 import MobileNetV3WithEmbedding, MobileNetV3WithEmbeddingV2, MobileNetV3_FaceRecognition




def plot_embeddings(model, dataloader, device, method='TSNE', n_components=2, perplexity=30, title="Embedding Distribution"):
    model.eval()  # Chuyển mô hình sang chế độ eval
    all_embeddings = []
    all_labels = []

    # Trích xuất embedding và nhãn
    with torch.no_grad():
        for anchor, positive, negative in dataloader:
            anchor = anchor.to(device)
            embeddings = model(anchor)
            all_embeddings.append(embeddings.cpu().numpy())  # Chuyển sang numpy
            all_labels.extend([0] * len(anchor))  # Gán nhãn cho ví dụ

    # Gộp lại thành mảng
    all_embeddings = torch.cat([torch.tensor(embed) for embed in all_embeddings], dim=0).numpy()

    # Giảm chiều
    if method == 'TSNE':
        reducer = TSNE(n_components=n_components, perplexity=perplexity, random_state=42)
    elif method == 'PCA':
        reducer = PCA(n_components=n_components)
    else:
        raise ValueError("method should be 'TSNE' or 'PCA'")
    
    reduced_embeddings = reducer.fit_transform(all_embeddings)

    # Vẽ biểu đồ
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x=reduced_embeddings[:, 0], 
        y=reduced_embeddings[:, 1], 
        hue=all_labels, 
        palette="tab10", 
        s=30
    )
    plt.title(title)
    plt.xlabel("Dimension 1")
    plt.ylabel("Dimension 2")
    plt.legend(title="Classes")
    plt.grid(True)
    plt.show()

# Ví dụ sử dụng:
dataset = "Face_Data"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MobileNetV3_FaceRecognition(embedding_dim=128).to(device)

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

print("Dataset Loading.........\n ")
dataset = FaceDataset(dataset, transform=transform)

print("Dataset Done.........\n ")

dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
plot_embeddings(model, dataloader, device, method='TSNE')
