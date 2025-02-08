import os
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

from xception import xception


class ImageDataset(Dataset):
    def __init__(self, folder: str, csv_file: str, prefix: str, transform=None):
        self.folder = folder
        self.csv_file = csv_file
        self.prefix = prefix
        self.filenames = [
            filename for filename in os.listdir(os.path.join(self.folder, self.prefix))
        ]
        self.labels = pd.read_csv(
            os.path.join(self.folder, self.csv_file), delimiter=","
        )
        self.filenames_to_labels_map = {}

        for _, row in self.labels.iterrows():
            self.filenames_to_labels_map[row[1]] = row[2]
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index: int):
        img_path = os.path.join(self.folder, self.prefix, self.filenames[index])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return (
            image,
            self.filenames_to_labels_map[self.prefix + "/" + self.filenames[index]],
        )


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.base_network = xception()
        self.num_classes = 2
        for param in self.base_network.parameters():
            param.requires_grad = False
        in_features = self.base_network.last_linear.in_features
        self.base_network.last_linear = nn.Linear(in_features, self.num_classes)
        conv4_params = self.base_network.conv4.parameters()
        last_linear_params = self.base_network.last_linear.parameters()

        for param in last_linear_params:
            param.requires_grad = True
        for param in conv4_params:
            param.requires_grad = True


if __name__ == "__main__":
    transform = transforms.Compose(
        [transforms.PILToTensor(), transforms.Resize((224, 224))]
    )

    train_dataset = ImageDataset(
        folder="data",
        csv_file="train.csv",
        prefix="train_data",
        transform=transform,
    )
    # test_dataset = ImageDataset(
    #    folder="data", csv_file="test.csv", prefix="test_data_v2", transform=transform
    # )

    dataloader = DataLoader(
        train_dataset, batch_size=128, shuffle=True, pin_memory=True
    )

    """
    state_dict = torch.load("./data/xception-b5690688.pth")

    for name, weights in state_dict.items():
        if "pointwise" in name:
            state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
    torch.save(state_dict, "xception-fixed.pth")
    """

    network = Network()
    learning_rate = 0.001
    optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
    train_logs = {
        "train_loss": [],
        "test_loss": [],
        "train_f1_score": [],
        "test_f1_score": [],
    }

    breakpoint()
