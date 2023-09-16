import os
from typing import Tuple

from PIL import Image
from torchvision.transforms import Compose, transforms
from torch.utils.data import Dataset, DataLoader


class FlowerDataset(Dataset):
    def __init__(self, img_dir: str, transform: Compose):
        self.img_dir = img_dir
        self.img_paths = sorted(os.listdir(self.img_dir))
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_paths[idx])

        img = Image.open(img_path)

        return self.transform(img)


def get_dataloader(data_dir: str, transform: Compose, batch_size: int) -> DataLoader:
    dataset = FlowerDataset(data_dir, transform=transform)

    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


def get_transform(img_size: Tuple[int, int]) -> Compose:
    return transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ColorJitter(brightness=0.05, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])
