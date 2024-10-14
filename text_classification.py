import numpy as np

from 

def load_data():
    X_train = np.load("data/data_train.npy")
    y_train = np.genfromtxt("data/label_train.csv", dtype=int, delimiter=",")[1:]
    X_test = np.load("data/data_test.npy")
    return X_train, y_train, X_test


def split_data(X, y, seed):
    indices = y[:, 0]
    np.random.seed(seed)
    np.random.shuffle(indices)
    train_indices = indices[: int(0.7 * len(indices))]
    val_indices = indices[int(0.7 * len(indices)) : int(0.9 * len(indices))]
    test_indices = indices[int(0.9 * len(indices)) :]
    X_train = X[train_indices]
    y_train = y[train_indices, 1]
    X_val = X[val_indices]
    y_val = y[val_indices, 1]
    X_test = X[test_indices]
    y_test = y[test_indices, 1]
    return X_train, y_train, X_val, y_val, X_test, y_test


def compute_metrics(y_true, y_pred, num_classes):
    # Initialize counts
    tp = np.zeros(num_classes)  # True positives
    fp = np.zeros(num_classes)  # False positives
    fn = np.zeros(num_classes)  # False negatives

    # Calculate TP, FP, FN for each class
    for i in range(len(y_true)):
        if y_pred[i] == y_true[i]:
            tp[y_true[i]] += 1  # Correctly predicted
        else:
            fp[y_pred[i]] += 1  # Incorrectly predicted as positive
            fn[y_true[i]] += 1  # Missed a positive

    # Calculate precision, recall, and F1 for each class
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1_score = np.zeros(num_classes)

    for i in range(num_classes):
        if tp[i] + fp[i] > 0:
            precision[i] = tp[i] / (tp[i] + fp[i])  # Precision for class i
        if tp[i] + fn[i] > 0:
            recall[i] = tp[i] / (tp[i] + fn[i])  # Recall for class i
        if precision[i] + recall[i] > 0:
            f1_score[i] = (
                2 * (precision[i] * recall[i]) / (precision[i] + recall[i])
            )  # F1 Score for class i

    # Compute macro averages
    macro_precision = np.mean(precision)
    macro_recall = np.mean(recall)
    macro_f1 = np.mean(f1_score)

    return macro_precision, macro_recall, macro_f1


class NaiveBayesClassifier:
    def __init__(self, alpha):
        self.alpha = alpha

    def fit(self, train_inputs, label_inputs):
        class1_prior = np.sum(label_inputs) / label_inputs.shape[0]
        log_priors = np.log([1 - class1_prior, class1_prior])
        mask = label_inputs == 0
        class0_features = train_inputs[mask]
        class1_features = train_inputs[~mask]
        feature_log_probs = np.log(
            [
                (np.sum(class0_features, axis=0) + self.alpha)
                / (np.sum(class0_features) + self.alpha * train_inputs.shape[1]),
                (np.sum(class1_features, axis=0) + self.alpha)
                / (np.sum(class1_features) + self.alpha * train_inputs.shape[1]),
            ]
        )
        log_probs = np.c_[feature_log_probs, log_priors]
        return log_probs

    def infer(self, test_inputs, w):
        return np.argmax(test_inputs @ w.T, axis=1)


def main():
    X_train, y_train, X_eval = load_data()
    X_train, y_train, X_val, y_val, X_test, y_test = split_data(X_train, y_train, 42)
    # Initialize arrays to store F1 score results for plotting
    alpha_values = np.arange(0, 10.1, 0.1)
    f1_train_values = []
    f1_val_values = []

    # Iterate over alpha values and compute macro F1 score for each
    for alpha in alpha_values:
        nbClf = NaiveBayesClassifier(alpha)
        log_probs = nbClf.fit(X_train, y_train)
        y_inferred_val = nbClf.infer(np.c_[X_val, np.ones(X_val.shape[0])], log_probs)
        y_inferred_train = nbClf.infer(
            np.c_[X_train, np.ones(X_train.shape[0])], log_probs
        )

        # Compute macro F1 scores for validation and training sets
        f1_val = compute_metrics(y_val, y_inferred_val, 2)
        f1_train = compute_metrics(y_train, y_inferred_train, 2)

        # Store the results for plotting
        f1_val_values.append(f1_val)
        f1_train_values.append(f1_train)

    # Plotting the macro F1 scores
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

    nbClf = NaiveBayesClassifier(0.1)
    log_probs = nbClf.fit(
        np.concatenate((X_train, X_val), axis=0),
        np.concatenate((y_train, y_val), axis=0),
    )
    y_inferred_test = nbClf.infer(np.c_[X_test, np.ones(X_test.shape[0])], log_probs)


if __name__ == "__main__":
    main()
