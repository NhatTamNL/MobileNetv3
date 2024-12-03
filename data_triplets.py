import os
import random
import torch
from torchvision import datasets, transforms
from PIL import Image
from torch.utils.data import Dataset


class FaceDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.image_paths = self._get_image_paths()

    def _get_image_paths(self):
        image_paths = []
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                image_paths.append((class_name, img_path))
        return image_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        class_name, anchor_img_path = self.image_paths[idx]
        anchor_img = Image.open(anchor_img_path).convert('RGB')

        if self.transform:
            anchor_img = self.transform(anchor_img)

        # Create positive (same class as anchor)
        positive_img_path = random.choice([img for img in self.image_paths if img[0] == class_name and img[1] != anchor_img_path])[1]
        positive_img = Image.open(positive_img_path).convert('RGB')

        if self.transform:
            positive_img = self.transform(positive_img)

        # Create negative (different class from anchor)
        negative_class = random.choice([cls for cls in self.classes if cls != class_name])
        negative_img_path = random.choice([img for img in self.image_paths if img[0] == negative_class])[1]
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
