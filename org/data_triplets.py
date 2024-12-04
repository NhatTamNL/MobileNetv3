import os
import random
import torch
from torchvision import datasets, transforms
from PIL import Image
from torch.utils.data import Dataset
from collections import defaultdict
import torch.nn.functional as F


# class FaceDataset(Dataset):
#     def __init__(self, root_dir, transform=None):
#         self.root_dir = root_dir
#         self.transform = transform
#         self.classes = os.listdir(root_dir)
#         self.image_paths = self._get_image_paths()

#     def _get_image_paths(self):
#         image_paths = []
#         for class_name in self.classes:
#             class_dir = os.path.join(self.root_dir, class_name)
#             for img_name in os.listdir(class_dir):
#                 img_path = os.path.join(class_dir, img_name)
#                 image_paths.append((class_name, img_path))
#         return image_paths

#     def __len__(self):
#         return len(self.image_paths)

#     def __getitem__(self, idx):
#         class_name, anchor_img_path = self.image_paths[idx]
#         anchor_img = Image.open(anchor_img_path).convert('RGB')

#         if self.transform:
#             anchor_img = self.transform(anchor_img)

#         # Create positive (same class as anchor)
#         positive_img_path = random.choice([img for img in self.image_paths if img[0] == class_name and img[1] != anchor_img_path])[1]
#         positive_img = Image.open(positive_img_path).convert('RGB')

#         if self.transform:
#             positive_img = self.transform(positive_img)

#         # Create negative (different class from anchor)
#         negative_class = random.choice([cls for cls in self.classes if cls != class_name])
#         negative_img_path = random.choice([img for img in self.image_paths if img[0] == negative_class])[1]
#         negative_img = Image.open(negative_img_path).convert('RGB')

#         if self.transform:
#             negative_img = self.transform(negative_img)
        
#         return anchor_img, positive_img, negative_img

class FaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = self._get_image_paths()
        self.class_to_images = self._get_class_to_images()

    def _get_image_paths(self):
        image_paths = []
        for class_name in os.listdir(self.root_dir):
            class_dir = os.path.join(self.root_dir, class_name)
            if not os.path.isdir(class_dir):  # Ignore non-directory files
                continue
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                image_paths.append((class_name, img_path))
        return image_paths

    def _get_class_to_images(self):
        class_to_images = {}
        for class_name, img_path in self.image_paths:
            if class_name not in class_to_images:
                class_to_images[class_name] = []
            class_to_images[class_name].append(img_path)
        return class_to_images

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        class_name, anchor_img_path = self.image_paths[idx]
        anchor_img = Image.open(anchor_img_path).convert('RGB')

        if self.transform:
            anchor_img = self.transform(anchor_img)

        # Create positive (same class as anchor)
        positive_candidates = [img for img in self.class_to_images[class_name] if img != anchor_img_path]
        if not positive_candidates:
            raise ValueError(f"Class '{class_name}' does not have enough images for positive samples.")
        positive_img_path = random.choice(positive_candidates)
        positive_img = Image.open(positive_img_path).convert('RGB')

        if self.transform:
            positive_img = self.transform(positive_img)

        # Create negative (different class from anchor)
        negative_class = random.choice([cls for cls in self.class_to_images.keys() if cls != class_name])
        negative_img_path = random.choice(self.class_to_images[negative_class])
        negative_img = Image.open(negative_img_path).convert('RGB')

        if self.transform:
            negative_img = self.transform(negative_img)
        
        return anchor_img, positive_img, negative_img
# Example usage
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
# ])
# # 
# dataset = FaceDataset(root_dir='facebank', transform=transform)


# class FaceDataset(Dataset):
#     def __init__(self, root_dir, model, transform=None):
#         self.root_dir = root_dir
#         self.transform = transform
#         self.classes = os.listdir(root_dir)
#         self.image_paths = self._get_image_paths()
#         self.class_to_images = defaultdict(list)
#         for class_name, img_path in self.image_paths:
#             self.class_to_images[class_name].append(img_path)
#         self.model = model  # Model for embedding extraction

#     def _get_image_paths(self):
#         image_paths = []
#         for class_name in self.classes:
#             class_dir = os.path.join(self.root_dir, class_name)
#             if not os.path.isdir(class_dir):
#                 continue
#             for img_name in os.listdir(class_dir):
#                 img_path = os.path.join(class_dir, img_name)
#                 if os.path.isfile(img_path):
#                     image_paths.append((class_name, img_path))
#         return image_paths

#     def __len__(self):
#         return len(self.image_paths)

#     def _load_image(self, img_path):
#         """ Load an image and apply transformations """
#         try:
#             img = Image.open(img_path).convert('RGB')
#             if self.transform:
#                 img = self.transform(img)  # Ensure it's converted to a tensor
#             return img
#         except Exception as e:
#             raise RuntimeError(f"Error loading image {img_path}: {e}")
    
#     def _get_embedding(self, img):
#         """ Get the embedding vector for an image using the model """
#         # Convert the image to a tensor if it's not already one
#         if not isinstance(img, torch.Tensor):
#             img = transforms.ToTensor()(img)  # Convert to tensor if it's a PIL Image
#         img = img.unsqueeze(0).to(torch.device('cuda'))  # Add batch dimension and move to GPU
#         with torch.no_grad():
#             embedding = self.model(img)
#         return embedding.squeeze(0)  # Remove batch dimension
    
#     def __getitem__(self, idx):
#         class_name, anchor_img_path = self.image_paths[idx]
#         anchor_img = self._load_image(anchor_img_path)

#         # Create positive sample
#         positive_img_path = random.choice([
#             img for img in self.class_to_images[class_name]
#             if img != anchor_img_path
#         ])
#         positive_img = self._load_image(positive_img_path)

#         # Hard negative mining
#         # Get the embeddings of the anchor and positive
#         anchor_embedding = self._get_embedding(anchor_img)
#         positive_embedding = self._get_embedding(positive_img)

#         # Calculate cosine similarity between anchor and positive (should be high)
#         anchor_positive_similarity = F.cosine_similarity(anchor_embedding, positive_embedding, dim=0).item()

#         # Choose a hard negative (closest negative that is not from the same class)
#         negative_class = random.choice([cls for cls in self.classes if cls != class_name])
#         negative_img_path = random.choice(self.class_to_images[negative_class])
#         negative_img = self._load_image(negative_img_path)
#         negative_embedding = self._get_embedding(negative_img)

#         # Calculate cosine similarity between anchor and negative (should be low)
#         anchor_negative_similarity = F.cosine_similarity(anchor_embedding, negative_embedding, dim=0).item()

#         return anchor_img, positive_img, negative_img, anchor_embedding, positive_embedding, negative_embedding


# class FaceDataset(Dataset):
#     def __init__(self, root_dir, transform=None, model=None):
#         self.root_dir = root_dir
#         self.transform = transform
#         self.model = model  # Thêm tham số model để tính embedding
#         self.classes = os.listdir(root_dir)
#         self.image_paths = self._get_image_paths()
#         self.embeddings = self._precompute_embeddings()

#     def _get_image_paths(self):
#         image_paths = []
#         for class_name in self.classes:
#             class_dir = os.path.join(self.root_dir, class_name)
#             for img_name in os.listdir(class_dir):
#                 img_path = os.path.join(class_dir, img_name)
#                 print(img_path, img_name,"Path\n")
#                 image_paths.append((class_name, img_path))  # Lưu (class_name, img_path)
#         return image_paths
    
#     def _precompute_embeddings(self):
#         """Tính trước tất cả embeddings cho từng ảnh."""
#         embeddings = {}
#         for class_name, img_path in self.image_paths:
#             img = Image.open(img_path).convert('RGB')
#             if self.transform:
#                 img = self.transform(img)
#             img = img.unsqueeze(0).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
#             with torch.no_grad():
#                 embeddings[img_path] = self.model(img)
#         return embeddings

#     def __len__(self):
#         return len(self.image_paths)

#     # def _get_embedding(self, img_path):
#     #     # Đảm bảo chỉ truyền img_path vào Image.open()
#     #     img = Image.open(img_path).convert('RGB')  # Mở ảnh từ đường dẫn img_path
#     #     if self.transform:
#     #         img = self.transform(img)
#     #     img = img.unsqueeze(0).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
#     #     embedding = self.model(img)
#     #     return embedding

#     # def __getitem__(self, idx):
#     #     # Lấy ảnh anchor và các embeddings của nó
#     #     class_name, anchor_img_path = self.image_paths[idx]
#     #     print(class_name, anchor_img_path, "Class Name\n")
#     #     anchor_embedding = self._get_embedding(anchor_img_path)
#     #     print(anchor_embedding,"anchor_embedding\n")


#     #     # Tạo positive sample (cùng lớp với anchor)
#     #     positive_img_path = random.choice([img for img in self.image_paths if img[0] == class_name and img[1] != anchor_img_path])[1]
#     #     positive_embedding = self._get_embedding(positive_img_path)
#     #     print(positive_embedding,"positive_embedding\n")

#     #     # Tính similarity giữa anchor và tất cả các negative samples
#     #     negative_similarities = []
#     #     negative_classes = [cls for cls in self.classes if cls != class_name]  # Các lớp khác class_name

#     #     for negative_class in negative_classes:
#     #         negative_img_paths = [img for img in self.image_paths if img[0] == negative_class]
#     #         for neg_img_path in negative_img_paths:
#     #             negative_embedding = self._get_embedding(neg_img_path[1])  # Truyền đúng đường dẫn ảnh
#     #             similarity = torch.nn.functional.cosine_similarity(anchor_embedding, negative_embedding)
#     #             print(similarity,"similarity\n")
#     #             negative_similarities.append((neg_img_path, similarity))

#     #     # Chọn hard negative sample (negative có similarity cao nhất với anchor)
#     #     hard_negative_img_path, _ = max(negative_similarities, key=lambda x: x[1])  # Chọn negative với similarity cao nhất
#     #     hard_negative_embedding = self._get_embedding(hard_negative_img_path[1])  # Truyền đúng đường dẫn ảnh
#     #     print(hard_negative_embedding,"hard_negative_embedding\n")
        

#     #     # Trả về anchor, positive, negative cùng với embeddings của chúng
#     #     return anchor_embedding, positive_embedding, hard_negative_embedding
#     def __getitem__(self, idx):
#         """Trả về anchor, positive, và hard negative embeddings."""
#         class_name, anchor_img_path = self.image_paths[idx]
#         anchor_embedding = self.embeddings[anchor_img_path]

#         # Positive sample (cùng lớp với anchor, khác ảnh)
#         positive_img_path = random.choice([img for img in self.image_paths if img[0] == class_name and img[1] != anchor_img_path])[1]
#         positive_embedding = self.embeddings[positive_img_path]

#         # Negative sample (chọn ngẫu nhiên từ các lớp khác)
#         negative_classes = [cls for cls in self.classes if cls != class_name]
#         negative_class = random.choice(negative_classes)
#         negative_img_path = random.choice([img for img in self.image_paths if img[0] == negative_class])[1]
#         negative_embedding = self.embeddings[negative_img_path]

#         return anchor_embedding, positive_embedding, negative_embedding