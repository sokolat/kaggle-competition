import argparse
import gc
import pickle
from datetime import datetime

import numpy as np
import timm
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train a machine learning model.")

    # Required arguments
    parser.add_argument("--data_path", type=str, help="Path to the dataset file.")
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


def read_data(path):
    # Load the pickle file
    with open(path, "rb") as f:
        data = pickle.load(f)
    # Access images and labels
    images = data["images"]
    labels = data["labels"]
    return images, labels


class VitBase16(nn.Module):
    def __init__(self, batch_size, lr, seed, epochs):
        super(VitBase16, self).__init__()
        self.seed = seed
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.labels = 4
        self.train = None
        self.val = None
        self.output_path = None
        self.model = timm.create_model("vit_base_patch16_224", pretrained=False)
        self.model.head = nn.Linear(self.model.head.in_features, self.labels)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def forward(self, X):
        X = self.model(X)
        return X

    def load_data(self, path, device):
        torch.manual_seed(self.seed)
        # Load the pickle file
        with open(path, "rb") as f:
            data = pickle.load(f)
        # Access images and labels
        images = torch.tensor(np.array(data["images"]), dtype=torch.float32).unsqueeze(
            1
        )
        labels = torch.tensor(np.array(data["labels"]), dtype=torch.float32)
        images.to(device)
        labels.to(device)
        X_train, X_val, y_train, y_val = train_test_split(
            images, labels, test_size=0.2, random_state=self.seed
        )
        X_train, X_val = self.process_images(X_train, X_val)
        print("ok")
        X_train.repeat(1, 3, 1, 1)
        X_val.repeat(1, 3, 1, 1)
        train_dataset = TensorDataset(X_train, y_train)
        val_dataset = TensorDataset(X_val, y_val)
        self.train = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=self.batch_size, shuffle=True
        )
        self.val = val_loader = torch.utils.data.DataLoader(
            dataset=val_dataset, batch_size=self.batch_size, shuffle=False
        )

    def process_images(self, train_images, val_images):
        # create image augmentations
        transforms_train = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                # transforms.RandomHorizontalFlip(p=0.3),
                # transforms.RandomVerticalFlip(p=0.3),
                # transforms.RandomResizedCrop(224),
                transforms.Normalize((0.5), (0.5)),
            ]
        )

        transforms_valid = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.Normalize((0.5), (0.5)),
            ]
        )
        return transforms_train(train_images), transforms_valid(val_images)

    def train_one_epoch(self, device):
        # keep track of training loss
        epoch_loss = 0.0
        epoch_accuracy = 0.0

        ###################
        # train the model #
        ###################
        self.model.train()
        for i, (data, target) in enumerate(self.train):
            data = data.to(device)
            target = target.to(device)

            # clear the gradients of all optimized variables
            self.optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = self.forward(data)
            # calculate the batch loss
            loss = self.criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # Calculate Accuracy
            accuracy = (output.argmax(dim=1) == target).float().mean()
            # update training loss and accuracy
            epoch_loss += loss
            epoch_accuracy += accuracy

            # perform a single optimization step (parameter update)
            optimizer.step()

        return epoch_loss / len(self.train), epoch_accuracy / len(self.train)

    def validate_one_epoch(self, device):
        # keep track of validation loss
        valid_loss = 0.0
        valid_accuracy = 0.0

        ######################
        # validate the model #
        ######################
        self.model.eval()
        for data, target in self.val:
            data = data.to(args.device)
            target = target.to(args.device)

            with torch.no_grad():
                # forward pass: compute predicted outputs by passing inputs to the model
                output = self.model(data)
                # calculate the batch loss
                loss = self.criterion(output, target)
                # Calculate Accuracy
                accuracy = (output.argmax(dim=1) == target).float().mean()
                # update average validation loss and accuracy
                valid_loss += loss
                valid_accuracy += accuracy

        return valid_loss / len(self.val), valid_accuracy / len(self.val)

    def fit(self, device, output_path):
        valid_loss_min = np.inf

        train_losses = []
        valid_losses = []
        train_accs = []
        valid_accs = []

        for epoch in range(1, self.epochs + 1):
            print(f"{'='*50}")
            print(f"EPOCH {epoch} - TRAINING...")
            train_loss, train_acc = self.train_one_epoch(device)
            print(
                f"\n\t[TRAIN] EPOCH {epoch} - LOSS: {train_loss}, ACCURACY: {train_acc}\n"
            )
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            gc.collect()

            print(f"EPOCH {epoch} - VALIDATING...")
            valid_loss, valid_acc = self.validate_one_epoch(device)
            xm.master_print(f"\t[VALID] LOSS: {valid_loss}, ACCURACY: {valid_acc}\n")
            valid_losses.append(valid_loss)
            valid_accs.append(valid_acc)
            gc.collect()

            # save model if validation loss has decreased
            if valid_loss <= valid_loss_min and epoch != 1:
                print(
                    "Validation loss decreased ({:.4f} --> {:.4f}).  Saving model ...".format(
                        valid_loss_min, valid_loss
                    )
                )
                torch.save(model.state_dict(), output_path)
                valid_loss_min = valid_loss

            return {
                "train_loss": train_losses,
                "valid_losses": valid_losses,
                "train_acc": train_accs,
                "valid_acc": valid_accs,
            }


def main():
    args = parse_args()
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
            print("ok")
            device = torch.device("mps")

    model = VitBase16(args.batch_size, args.learning_rate, args.seed, args.epochs)
    model.to(device)
    print(next(model.parameters()).device)
    model.load_data(args.data_path, device)

    print(f"INITIALIZING TRAINING ON MPS GPU")
    start_time = datetime.now()
    print(f"Start Time: {start_time}")

    logs = model.fit(device, args.output_path)

    print(f"Execution time: {datetime.now() - start_time}")

    torch.save(
        model.state_dict, f'model_5e_{datetime.now().strftime("%Y%m%d-%H%M")}.pth'
    )


if __name__ == "__main__":
    main()
