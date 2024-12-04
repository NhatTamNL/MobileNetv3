import os
import random
from torchvision import datasets, transforms
from PIL import Image
from torch.utils.data import Dataset

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

# dataset = FaceDataset(root_dir='facebank', transform=transform)