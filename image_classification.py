import argparse
import pickle

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description="Train a machine learning model.")

    # Required arguments
    parser.add_argument("--data_path", type=str, help="Path to the dataset file.")
    parser.add_argument(
        "--output_path", type=str, help="Directory to save the trained model and logs."
    )

    # Optional arguments with default values
    parser.add_argument(
        "--model",
        type=str,
        default="hard_parzen",
        choices=["hard_parzen"],
        help="Type of model to train (default: random_forest).",
    )
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


# Function to split the dataset into training, validation, and test sets
def split_data(X, y, seed):
    indices = np.arange(0, len(y))  # Get the indices of labels
    np.random.seed(seed)  # Set random seed for reproducibility
    np.random.shuffle(indices)  # Shuffle the indices randomly
    # Split indices into 80% training, 20% validation
    train_indices = indices[: int(0.8 * len(indices))]
    val_indices = indices[int(0.8 * len(indices)) :]
    # Split features and labels according to the indices
    X_train = X[train_indices]
    y_train = y[train_indices]
    X_val = X[val_indices]
    y_val = y[val_indices]
    return X_train, y_train, X_val, y_val


def main():
    args = parse_args()
    X, y = read_data(args.data_path)
    X_train, y_train, X_val, y_val = split_data(np.array(X), np.array(y), args.seed)


if __name__ == "__main__":
    main()
