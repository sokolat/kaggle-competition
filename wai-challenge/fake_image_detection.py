import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.models import DenseNet121_Weights, densenet121


class ImageDataset(Dataset):
    def __init__(self, folder: str, transform=None):
        self.folder = folder
        self.image_files = [filename for filename in os.listdir(folder)]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        img_path = os.path.join(self.folder, self.image_files[index])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.base_model = densenet121(weights=DenseNet121_Weights.IMAGENET1K_V1)
        self.num_classes = 2
        for param in self.base_model.parameters():
            param.requires_grad = False
        in_features = self.base_model.classifier.in_features
        self.base_model.classifier = nn.Linear(in_features, self.num_classes)


if __name__ == "__main__":
    transform = transforms.Compose(
        [transforms.PILToTensor(), transforms.Resize((224, 224))]
    )
    dataset = ImageDataset(folder="./train_data", transform=transform)
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True, pin_memory=True)
    model = Model()
    print(model.base_model)
