from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.ensemble import (
    BaggingClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import SVC, LinearSVC


# Function to load training and testing data
def load_data():
    data_train = np.load("data/data_train.npy")  # Load training features from .npy file
    label_train = np.genfromtxt("data/label_train.csv", dtype=int, delimiter=",")[
        1:
    ]  # Load training labels from CSV, skipping the header
    data_test = np.load("data/data_test.npy")  # Load testing features from .npy file
    return data_train, label_train, data_test


def get_wordnet_pos(word):
    tag = pos_tag([word])[0][1][0].upper()
    tag_dict = {
        "J": wordnet.ADJ,
        "N": wordnet.NOUN,
        "V": wordnet.VERB,
        "R": wordnet.ADV,
    }
    return tag_dict.get(tag, wordnet.NOUN)


def main():
    wnl = WordNetLemmatizer()
    vocab_maps = np.load("data/vocab_map.npy", allow_pickle=True)
    print(vocab_maps.shape)
    lemmatized_vocabs = [
        wnl.lemmatize(word, get_wordnet_pos(word)) for word in vocab_maps
    ]
    indices_to_remove = []

    index_map = defaultdict(list)
    for index, value in enumerate(lemmatized_vocabs):
        if value in stopwords.words("english"):
            indices_to_remove.append(index)
        index_map[value].append(index)
    matching_indices = list(index_map.values())
    data_train, label_train, data_test = load_data()
    for indices in matching_indices:
        main_index = indices[0]
        for assoc_index in indices[1:]:
            data_train[:, main_index] += data_train[:, assoc_index]
            data_test[:, main_index] += data_test[:, assoc_index]
            indices_to_remove.append(assoc_index)
    X_train = np.delete(data_train, indices_to_remove, 1)
    X_test = np.delete(data_test, indices_to_remove, 1)
    print(X_train.shape)
    print(X_train.shape)
    # Split the data into training, validation, and test sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, label_train[:, 1], test_size=0.2, random_state=42
    )
    transformer = TfidfTransformer()
    X_train_tf_idf = transformer.fit_transform(X_train).toarray()
    X_val_tf_idf = transformer.fit_transform(X_val).toarray()
    alpha_vals = np.arange(0, 11, 0.1)
    train_f1_scores = []
    validation_f1_scores = []
    for alpha in alpha_vals:
        clf = ComplementNB(alpha=alpha)
        clf.fit(X_train, y_train)
        y_val_pred = clf.predict(X_val_tf_idf)
        y_train_pred = clf.predict(X_train_tf_idf)
        train_f1_score = f1_score(y_train, y_train_pred, average="macro")
        validation_f1_score = f1_score(y_val, y_val_pred, average="macro")
        print(
            f"alpha: {alpha}, Train F1 Score: {train_f1_score:.3f}, Validation F1 Score: {validation_f1_score:.3f}"
        )
        train_f1_scores.append(train_f1_score)
        validation_f1_scores.append(validation_f1_score)

    plt.figure(figsize=(8, 6))
    plt.plot(alpha_vals, train_f1_scores, label="Train F1 Score", marker="o")
    plt.plot(alpha_vals, validation_f1_scores, label="Validation F1 Score", marker="o")

    plt.title("Train vs Validation F1 Score for Different Alpha Values")
    plt.xlabel("Alpha")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.grid(True)
    plt.show()
    y_pred = clf.predict(X_test)
    indices = np.arange(len(y_pred))  # Generate indices for the predictions
    # Write predictions to a CSV file
    with open("pred.csv", "w") as f:
        f.write("ID,label\n")
        for each_id, label in zip(indices, y_pred):
            line = f"{each_id},{label}\n"
            f.write(line)


# Run the main function
if __name__ == "__main__":
    main()
