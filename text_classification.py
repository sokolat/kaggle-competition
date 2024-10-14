import numpy as np
from matplotlib import pyplot as plt


# Function to load training and testing data
def load_data():
    X_train = np.load("data/data_train.npy")  # Load training features from .npy file
    y_train = np.genfromtxt("data/label_train.csv", dtype=int, delimiter=",")[
        1:
    ]  # Load training labels from CSV, skipping the header
    X_test = np.load("data/data_test.npy")  # Load testing features from .npy file
    return X_train, y_train, X_test


# Function to split the dataset into training, validation, and test sets
def split_data(X, y, seed):
    indices = y[:, 0]  # Get the indices of labels
    np.random.seed(seed)  # Set random seed for reproducibility
    np.random.shuffle(indices)  # Shuffle the indices randomly
    # Split indices into 70% training, 20% validation, and 10% testing
    train_indices = indices[: int(0.7 * len(indices))]
    val_indices = indices[int(0.7 * len(indices)) : int(0.9 * len(indices))]
    test_indices = indices[int(0.9 * len(indices)) :]
    # Split features and labels according to the indices
    X_train = X[train_indices]
    y_train = y[train_indices, 1]
    X_val = X[val_indices]
    y_val = y[val_indices, 1]
    X_test = X[test_indices]
    y_test = y[test_indices, 1]
    return X_train, y_train, X_val, y_val, X_test, y_test


# Function to compute the macro F1 score given true and predicted labels
def compute_metrics(y_true, y_pred):
    tp = np.zeros(2)  # True positives for each class
    fp = np.zeros(2)  # False positives for each class
    fn = np.zeros(2)  # False negatives for each class

    # Calculate TP, FP, and FN for each prediction
    for i in range(len(y_true)):
        if y_pred[i] == y_true[i]:
            tp[y_true[i]] += 1
        else:
            fp[y_pred[i]] += 1
            fn[y_true[i]] += 1

    precision = np.zeros(2)  # Precision for each class
    recall = np.zeros(2)  # Recall for each class
    f1_score = np.zeros(2)  # F1 score for each class

    # Calculate precision, recall, and F1 score for each class
    for i in range(2):
        if tp[i] + fp[i] > 0:
            precision[i] = tp[i] / (tp[i] + fp[i])
        if tp[i] + fn[i] > 0:
            recall[i] = tp[i] / (tp[i] + fn[i])
        if precision[i] + recall[i] > 0:
            f1_score[i] = 2 * (precision[i] * recall[i]) / (precision[i] + recall[i])

    macro_f1 = np.mean(f1_score)  # Calculate the macro F1 score (average of F1 scores)
    return macro_f1


# Naive Bayes classifier class
class NaiveBayesClassifier:
    def __init__(self, alpha):
        self.alpha = alpha  # Smoothing parameter

    # Function to fit the model using training data
    def fit(self, train_inputs, label_inputs):
        class1_prior = (
            np.sum(label_inputs) / label_inputs.shape[0]
        )  # Prior probability of class 1
        log_priors = np.log([1 - class1_prior, class1_prior])  # Log of class priors
        mask = label_inputs == 0  # Mask for class 0
        class0_features = train_inputs[mask]  # Features for class 0
        class1_features = train_inputs[~mask]  # Features for class 1
        # Compute log probabilities for each feature conditioned on the class
        feature_log_probs = np.log(
            [
                (np.sum(class0_features, axis=0) + self.alpha)
                / (np.sum(class0_features) + self.alpha * train_inputs.shape[1]),
                (np.sum(class1_features, axis=0) + self.alpha)
                / (np.sum(class1_features) + self.alpha * train_inputs.shape[1]),
            ]
        )
        log_probs = np.c_[
            feature_log_probs, log_priors
        ]  # Combine feature log probs and priors
        return log_probs

    # Function to make predictions using the model
    def infer(self, test_inputs, w):
        return np.argmax(
            test_inputs @ w.T, axis=1
        )  # Predict class with the highest probability


# Main function to execute the workflow
def main():
    # Load the training and evaluation data
    X_train, y_train, X_eval = load_data()
    # Split the data into training, validation, and test sets
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X_train, y_train, 42)
    alpha_values = np.arange(0, 10.1, 0.1)  # Range of alpha values to test
    f1_train_values = []  # Store F1 scores for training data
    f1_val_values = []  # Store F1 scores for validation data

    # Loop over alpha values and evaluate the model
    for alpha in alpha_values:
        nbClf = NaiveBayesClassifier(alpha)  # Initialize Naive Bayes classifier
        log_probs = nbClf.fit(X_train, y_train)  # Train the model
        # Predict labels for validation and training sets
        y_inferred_val = nbClf.infer(np.c_[X_val, np.ones(X_val.shape[0])], log_probs)
        y_inferred_train = nbClf.infer(
            np.c_[X_train, np.ones(X_train.shape[0])], log_probs
        )

        # Compute F1 scores
        f1_val = compute_metrics(y_val, y_inferred_val)
        f1_train = compute_metrics(y_train, y_inferred_train)

        # Store the F1 scores
        f1_val_values.append(f1_val)
        f1_train_values.append(f1_train)

    # Plotting the macro F1 scores (commented out)
    """
    plt.figure(figsize=(10, 6))
    plt.plot(
        alpha_values,
        f1_val_values,
        label="F1 Score (Validation)",
        linestyle="-",
        marker="o",
    )
    plt.plot(
        alpha_values,
        f1_train_values,
        label="F1 Score (Training)",
        linestyle="-",
        marker="o",
    )
    plt.xlabel("Alpha")
    plt.ylabel("Macro F1 Score")
    plt.title("Macro F1 Score vs Alpha")
    plt.legend()
    plt.grid(True)
    plt.show()
    """

    # Find the optimal alpha value based on the highest validation F1 score
    alpha_opt = alpha_values[np.argmax(f1_val_values)]

    # Retrain the model using the optimal alpha and the entire dataset
    nbClf = NaiveBayesClassifier(alpha_opt)
    log_probs = nbClf.fit(
        np.concatenate((X_train, X_val, X_test), axis=0),
        np.concatenate((y_train, y_val, y_test), axis=0),
    )
    # Infer labels for the evaluation set
    y_inferred_test = nbClf.infer(np.c_[X_eval, np.ones(X_eval.shape[0])], log_probs)
    indices = np.arange(len(y_inferred_test))  # Generate indices for the predictions
    # Write predictions to a CSV file
    with open("pred.csv", "w") as f:
        f.write("ID,label\n")
        for each_id, label in zip(indices, y_inferred_test):
            line = f"{each_id},{label}\n"
            f.write(line)


# Run the main function
if __name__ == "__main__":
    main()
