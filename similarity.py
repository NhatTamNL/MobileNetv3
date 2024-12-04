import torch
from utils import get_embedding, compare_embeddings
from mobileNetv3 import MobileNetV3WithEmbedding, MobileNetV3_FaceRecognition

# Thiết lập mô hình
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = MobileNetV3_FaceRecognition(embedding_dim=128).to(device)
model.load_state_dict(torch.load('models_test/mobilenetv3_epoch35.pth'))  # Tải mô hình đã huấn luyện

# Lấy embedding của hai ảnh khuôn mặt để so sánh
embedding1 = get_embedding('data_test/' + '44.jpg', model, device)
embedding2 = get_embedding('data_test/' + 'Joey00022.png', model, device)

# So sánh độ tương đồng giữa hai embedding
similarity = compare_embeddings(embedding1, embedding2)
print(similarity)

# Quyết định liệu đây có phải là cùng một người hay không
threshold = 0.7  # Ngưỡng cosine similarity
if similarity > threshold:
    print("Hai ảnh là của cùng một người")
else:
    print("Hai ảnh là của người khác nhau")
