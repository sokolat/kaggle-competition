import argparse
import os
from datetime import datetime
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torcheval.metrics.functional import multiclass_f1_score
from torchvision.transforms import v2
from tqdm import tqdm

from xception import xception


def parse_args():
    parser = argparse.ArgumentParser(description="Train a machine learning model.")

    # Required arguments
    parser.add_argument(
        "--output_path", type=str, help="Directory to save the trained model and logs."
    )
    # Optional arguments with default values
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.01,
        help="Learning rate for training (default: 0.01).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs (default: 10).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for training (default: 32).",
    )

    # Boolean flags
    parser.add_argument(
        "--verbose", action="store_true", help="Enable verbose logging."
    )
    parser.add_argument(
        "--use_gpu", action="store_true", help="Use GPU for training if available."
    )
    parser.add_argument("--device", type=str, help="GPU device for heavy computation")

    parser.add_argument("--seed", type=int, default=42, help="Seed for reproducibility")

    return parser.parse_args()


class TrainImageDataset(Dataset):
    def __init__(self, transform=None):
        self.filenames = [
            filename for filename in os.listdir(os.path.join("data/train_data"))
        ]
        self.labels = pd.read_csv(os.path.join("data/train.csv"), delimiter=",")
        self.filenames_to_labels_map = {}

        for _, row in self.labels.iterrows():
            self.filenames_to_labels_map[row[1]] = row[2]
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index: int):
        img_path = os.path.join("data/train_data", self.filenames[index])
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return (
            image,
            self.filenames_to_labels_map["train_data" + "/" + self.filenames[index]],
        )


class TestImageDataset(Dataset):
    def __init__(self, transform=None):
        self.filenames = [
            filename for filename in os.listdir(os.path.join("data/test_data_v2"))
        ]
        self.labels = pd.read_csv("data/test.csv", delimiter=",")

        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index: int):
        img_path = os.path.join(
            "data/test_data_v2",
            self.filenames[index],
        )
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return (image, self.filenames[index])


class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.base_network = xception()
        self.num_classes = 2
        for param in self.base_network.parameters():
            param.requires_grad = False
        in_features = self.base_network.last_linear.in_features
        self.base_network.last_linear = nn.Sequential(
            nn.Dropout(p=0.5), nn.Linear(in_features, self.num_classes)
        )
        conv4_params = self.base_network.conv4.parameters()
        last_linear_params = self.base_network.last_linear.parameters()

        for param in last_linear_params:
            param.requires_grad = True
        for param in conv4_params:
            param.requires_grad = True


if __name__ == "__main__":
    args = parse_args()

    torch.manual_seed(args.seed)

    device = None
    if args.device == "mps":
        if not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print(
                    "MPS not available because the current PyTorch install was not "
                    "built with MPS enabled."
                )
            else:
                print(
                    "MPS not available because the current MacOS version is not 12.3+ "
                    "and/or you do not have an MPS-enabled device on this machine."
                )
        else:
            device = torch.device("mps")

    elif args.device == "cuda":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"INITIALIZING TRAINING ON {args.device.upper()} GPU")
    start_time = datetime.now()
    print(f"Start Time: {start_time}")

    transform = v2.Compose(
        [
            v2.ToTensor(),
            v2.Resize(333),
            v2.CenterCrop(299),
            v2.RandomHorizontalFlip(p=0.5),
            v2.RandomVerticalFlip(p=0.5),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    dataset = TrainImageDataset(transform=transform)

    validation_split = 0.2
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    np.random.seed(args.seed)
    train_indices, val_indices = indices[split:], indices[:split]

    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True
    )
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True)

    model = Network().base_network.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.learning_rate, weight_decay=0.00001
    )
    criterion = nn.CrossEntropyLoss()

    valid_loss_min = np.inf
    train_losses = []
    valid_losses = []
    train_f1_scores = []
    valid_f1_scores = []
    for epoch in tqdm(range(1, args.epochs + 1)):
        print(f"{'='*50}")
        print(f"EPOCH {epoch} - TRAINING...")

        epoch_loss = 0.0
        epoch_f1_score = 0.0

        model.train()
        for data, target in tqdm(train_dataloader):
            data = data.to(device, dtype=torch.float32)
            target = target.to(device, dtype=torch.long)

            optimizer.zero_grad()

            output = model(data)
            loss = criterion(output, target)

            loss.backward()

            # f_score = f1_score(target.cpu(), output.cpu().argmax(dim=1))
            f_score = multiclass_f1_score(output, target, num_classes=2)
            epoch_loss += loss.item()
            epoch_f1_score += f_score.item()

            optimizer.step()

        train_loss, train_f1_score = epoch_loss / len(
            train_dataloader
        ), epoch_f1_score / len(train_dataloader)
        print(
            f"\n\t[TRAIN] EPOCH {epoch} - LOSS: {train_loss}, F1-SCORE: {train_f1_score}\n"
        )
        train_losses.append(train_loss)
        train_f1_scores.append(train_f1_score)

        print(f"EPOCH {epoch} - VALIDATING...")

        valid_loss = 0.0
        valid_f1_score = 0.0

        model.eval()

        for data, target in tqdm(val_dataloader):
            data = data.to(device, dtype=torch.float32)
            target = target.to(device, dtype=torch.long)

            with torch.no_grad():
                output = model(data)
                loss = criterion(output, target)
                # f_score = f1_score(target.cpu(), output.cpu().argmax(dim=1))
                f_score = multiclass_f1_score(output, target, num_classes=2)
                valid_loss += loss.item()
                valid_f1_score += f_score.item()

        val_loss, val_f1_socre = valid_loss / len(val_dataloader), valid_f1_score / len(
            val_dataloader
        )
        print(f"\t[VALID] LOSS: {val_loss}, F1-SCORE: {val_f1_socre}\n")
        valid_losses.append(val_loss)
        valid_f1_scores.append(val_f1_socre)

        if val_loss <= valid_loss_min and epoch != 1:
            print(
                "Validation loss decreased ({:.4f} --> {:.4f}).  Saving model ...".format(
                    valid_loss_min, val_loss
                )
            )
            torch.save(
                model.state_dict(),
                f"{args.output_path}.pth",
            )
            valid_loss_min = val_loss

    logs = {
        "train_losses": train_losses,
        "valid_losses": valid_losses,
        "train_f1_scores": train_f1_scores,
        "valid_f1_scores": valid_f1_scores,
    }

    print(f"Execution time: {datetime.now() - start_time}")

    torch.save(
        model.state_dict,
        f'model_{datetime.now().strftime("%Y%m%d-%H%M")}.pth',
    )
    """
    test_transform = v2.Compose(
        v2.ToTensor(),
        v2.Resize(333),
        v2.CenterCrop(299),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    )
    """
    test_dataset = TestImageDataset(transform)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)

    model.load_state_dict(torch.load("./data/best_model_params.pth"))

    model.eval()
    predictions = {}

    with torch.no_grad():
        for images, filenames in tqdm(test_dataloader):
            images = images.to(device, dtype=torch.float32)

            outputs = model(images)
            probs = F.softmax(outputs, dim=1)
            preds = torch.argmax(probs, dim=1)

            for filename, pred in zip(filenames, preds.cpu().numpy()):
                predictions[filename] = pred

    target = pd.read_csv("./data/test.csv")
    labels = pd.DataFrame(list(predictions.items()), columns=["id", "label"])
    labels["id"] = "test_data_v2" + labels["id"]
    merged = pd.merge(target, labels, on="id", how="inner")
    merged.index = merged.index + 1
    merged.to_csv("pred.csv", index=False)

    train_losses = logs["train_losses"]
    valid_losses = logs["valid_losses"]
    train_f1_scores = logs["train_f1_scores"]
    valid_f1_scores = logs["valid_f1_scores"]

    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.plot(
        np.arange(1, len(train_losses) + 1),
        train_losses,
        label="train loss",
        marker="o",
    )
    plt.plot(
        np.arange(1, len(valid_losses) + 1),
        valid_losses,
        label="validation loss",
        marker="o",
    )
    plt.title("loss: train vs validation")
    plt.xlabel("epochs")
    plt.ylabel("loss")
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(
        np.arange(1, len(train_f1_scores) + 1),
        train_f1_scores,
        label="train F1-score",
        marker="o",
    )
    plt.plot(
        np.arange(1, len(valid_f1_scores) + 1),
        valid_f1_scores,
        label="validation F1-score",
        marker="o",
    )
    plt.title("Loss: Train vs Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()

    plt.subplot(1, 2, 2)
    plt.plot(
        np.arange(1, args.epochs + 1),
        train_f1_scores,
        label="Train  F1-score",
        marker="o",
    )
    plt.plot(
        np.arange(1, args.epochs + 1),
        valid_f1_scores,
        label="Validation F1-score",
        marker="o",
    )
    plt.title("F1-score: Train vs Validation")
    plt.xlabel("Epochs")
    plt.ylabel("F1-score")
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.savefig("train_val_metrics.png")
    plt.show()
