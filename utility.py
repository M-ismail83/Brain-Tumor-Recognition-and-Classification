import torchvision.transforms as transforms
import os
from torch.utils.data import Dataset
from PIL import Image


class Utility:

    def __init__(self, img_size, s=1):
        color_jitter = transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )

        blur = transforms.GaussianBlur((3, 3), (0.1, 2.0))

        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=img_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomApply([blur], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        return self.train_transform(x), self.train_transform(x)

    def image_visualize(self):
        pass  # Will put gradCAM later


class CustomDataset(Dataset):
    def __init__(self, folder_path, transform):
        self.folder_path = folder_path
        self.transform = transform
        self.image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        if len(self.image_files) == 0:
            raise ValueError(f"No images found in folder: {folder_path}")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.folder_path, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')  # Ensure 3 channels
        xi, xj = self.transform(image)

        return xi, xj


class HParams:
    def __init__(self, epochs=100, embedding_size=128, lr=1e-2, weight_decay=1e-6):
        self.batch_size = 20
        self.temperature = 0.07
        self.img_size = 224
        self.gradient_accumulation_steps = 1
        self.epochs = epochs
        self.embedding_size = embedding_size
        self.lr = lr
        self.weight_decay = weight_decay
        self.momentum = 0.9
        self.checkpoint_path = "saved_models/SimCLR_Resnet18_Adam.ckpt"


class FineTuneDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.samples = []
        self.classes = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
        if not self.classes:
            raise ValueError(f"No class subfolders found in {root_dir}")
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.num_classes = len(self.classes)

        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.samples.append((os.path.join(cls_dir, img_name), self.class_to_idx[cls_name]))

        if not self.samples:
            raise ValueError(f"No images found in {root_dir} subdirectories.")
        print(f"Found {len(self.samples)} images in {self.num_classes} classes for directory {root_dir}.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


def get_finetune_transforms(img_size=224, is_train=True):
    if is_train:
        return transforms.Compose([
            transforms.RandomResizedCrop(img_size, scale=(0.2, 1.0)), # SimCLR uses 0.08, can be gentler for finetune
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1), # Optional, can be milder
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else: # Validation/Test
        return transforms.Compose([
            transforms.Resize(img_size + 32), # e.g., 256 if img_size is 224
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])