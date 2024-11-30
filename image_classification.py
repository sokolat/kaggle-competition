import argparse
import gc
import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import tqdm
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train a machine learning model.")

    # Required arguments
    parser.add_argument(
        "--train_data_path", type=str, help="Path to the train dataset file."
    )
    parser.add_argument(
        "--test_data_path", type=str, help="Path to the test dataset file."
    )
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


class Cnn:
    def __init__(self, batch_size, lr, seed, epochs, device):
        self.seed = seed
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.train = None
        self.val = None
        self.test = None
        self.output_path = None
        self.device = device
        self.model = nn.Sequential(
            nn.Conv2d(1, 16, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(128, 4),
        ).to(device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

    def load_data(self, train_path, test_path):
        torch.manual_seed(self.seed)
        # Load the pickle file
        if train_path:
            with open(train_path, "rb") as f:
                data = pickle.load(f)
            # Access images and labels
            images = torch.tensor(
                np.array(data["images"]), dtype=torch.float32
            ).unsqueeze(1)
            labels = torch.tensor(np.array(data["labels"]), dtype=torch.int64)
            X_train, X_val, y_train, y_val = train_test_split(
                images, labels, test_size=0.2, random_state=self.seed
            )
            train_dataset = TensorDataset(X_train, y_train)
            val_dataset = TensorDataset(X_val, y_val)
            self.train = torch.utils.data.DataLoader(
                dataset=train_dataset, batch_size=self.batch_size, shuffle=True
            )
            self.val = torch.utils.data.DataLoader(
                dataset=val_dataset, batch_size=self.batch_size, shuffle=False
            )
        if test_path:
            with open(test_path, "rb") as f:
                data = pickle.load(f)
            # Access images and labels
            self.test = (
                torch.tensor(np.array(data["images"]), dtype=torch.float32)
                .unsqueeze(1)
                .to(self.device)
            )

    def train_one_epoch(self):
        # keep track of training loss
        epoch_loss = 0.0
        epoch_accuracy = 0.0

        ###################
        # train the model #
        ###################
        self.model.train()
        for i, (data, target) in enumerate(self.train):
            data = data.to(self.device, dtype=torch.float32)
            target = target.to(self.device, dtype=torch.long)

            # clear the gradients of all optimized variables
            self.optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = self.model(data)
            # calculate the batch loss
            loss = self.criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # Calculate Accuracy
            accuracy = (output.argmax(dim=1) == target).float().mean()
            # update training loss and accuracy
            epoch_loss += loss.item()
            epoch_accuracy += accuracy

            # perform a single optimization step (parameter update)
            self.optimizer.step()

        return epoch_loss / len(self.train), epoch_accuracy / len(self.train)

    def validate_one_epoch(self):
        # keep track of validation loss
        valid_loss = 0.0
        valid_accuracy = 0.0

        ######################
        # validate the model #
        ######################
        self.model.eval()
        for data, target in self.val:
            data = data.to(self.device, dtype=torch.float32)
            target = target.to(self.device, dtype=torch.long)

            with torch.no_grad():
                # forward pass: compute predicted outputs by passing inputs to the model
                output = self.model(data)
                # calculate the batch loss
                loss = self.criterion(output, target)
                # Calculate Accuracy
                accuracy = (output.argmax(dim=1) == target).float().mean()
                # update average validation loss and accuracy
                valid_loss += loss.item()
                valid_accuracy += accuracy

        return valid_loss / len(self.val), valid_accuracy / len(self.val)

    def fit(self, output_path):
        valid_loss_min = np.inf

        train_losses = []
        valid_losses = []
        train_accs = []
        valid_accs = []

        for epoch in range(1, self.epochs + 1):
            print(f"{'='*50}")
            print(f"EPOCH {epoch} - TRAINING...")
            train_loss, train_acc = self.train_one_epoch()
            print(
                f"\n\t[TRAIN] EPOCH {epoch} - LOSS: {train_loss}, ACCURACY: {train_acc}\n"
            )
            train_losses.append(train_loss)
            train_accs.append(train_acc)
            gc.collect()

            print(f"EPOCH {epoch} - VALIDATING...")
            valid_loss, valid_acc = self.validate_one_epoch()
            print(f"\t[VALID] LOSS: {valid_loss}, ACCURACY: {valid_acc}\n")
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
                torch.save(
                    self.model.state_dict(),
                    f"{output_path}.pth",
                )
                valid_loss_min = valid_loss

        return {
            "train_losses": train_losses,
            "valid_losses": valid_losses,
            "train_accs": train_accs,
            "valid_accs": valid_accs,
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
            device = torch.device("mps")

    cnn = Cnn(args.batch_size, args.learning_rate, args.seed, args.epochs, device)
    cnn.load_data(args.train_data_path, args.test_data_path)

    print(f"INITIALIZING TRAINING ON MPS GPU")
    start_time = datetime.now()
    print(f"Start Time: {start_time}")

    logs = cnn.fit(args.output_path)

    # Convert losses and accuracies to NumPy arrays
    train_losses = logs["train_losses"]
    valid_losses = logs["valid_losses"]
    train_accs = logs["train_accs"]
    valid_accs = logs["valid_accs"]

    # create the figure
    plt.figure(figsize=(10, 5))

    # plot losses
    plt.subplot(1, 2, 1)  # 1 row, 2 columns, 1st plot
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

    # plot accuracies
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd plot
    plt.plot(
        np.arange(1, len(train_accs) + 1),
        train_accs,
        label="train accuracy",
        marker="o",
    )
    plt.plot(
        np.arange(1, len(valid_accs) + 1),
        valid_accs,
        label="validation accuracy",
        marker="o",
    )
    plt.title("Loss: Train vs Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid()

    # Plot Accuracies
    plt.subplot(1, 2, 2)  # 1 row, 2 columns, 2nd plot
    plt.plot(
        np.arange(1, args.epochs + 1), train_accs, label="Train Accuracy", marker="o"
    )
    plt.plot(
        np.arange(1, args.epochs + 1),
        valid_accs,
        label="Validation Accuracy",
        marker="o",
    )
    plt.title("Accuracy: Train vs Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.grid()

    # Display the plots
    plt.tight_layout()
    plt.savefig("train_val_metrics.png")
    plt.show()

    print(f"Execution time: {datetime.now() - start_time}")

    torch.save(
        cnn.model.state_dict, f'model_{datetime.now().strftime("%Y%m%d-%H%M")}.pth'
    )

    # Load the model state
    cnn.model.load_state_dict(torch.load("./data/out/best_model_params.pth"))
    cnn.model.eval()
    with torch.no_grad():
        preds = cnn.model(cnn.test).argmax(dim=1)

    # Write predictions to a CSV file
    with open("pred.csv", "w") as f:
        f.write("ID, Class\n")
        for each_id, label in enumerate(preds):
            line = f"{each_id+1},{label}\n"
            f.write(line)


if __name__ == "__main__":
    main()
