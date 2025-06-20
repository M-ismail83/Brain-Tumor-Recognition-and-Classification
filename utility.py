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
