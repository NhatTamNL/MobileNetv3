import torch
from utils import get_embedding, compare_embeddings
from mobileNetv3 import MobileNetV3WithEmbedding

# Thiết lập mô hình
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MobileNetV3WithEmbedding(embedding_dim=128).to(device)
model.load_state_dict(torch.load('mobilenetv3_epoch20.pth'))  # Tải mô hình đã huấn luyện

# Lấy embedding của hai ảnh khuôn mặt để so sánh
embedding1 = get_embedding('A.Phuc_3.jpg', model, device)
embedding2 = get_embedding('Trong_6.jpg', model, device)

# So sánh độ tương đồng giữa hai embedding
similarity = compare_embeddings(embedding1, embedding2)

# Quyết định liệu đây có phải là cùng một người hay không
threshold = 0.8  # Ngưỡng cosine similarity
if similarity > threshold:
    print("Hai ảnh là của cùng một người")
else:
    print("Hai ảnh là của người khác nhau")
