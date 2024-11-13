from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
from nltk import pos_tag
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import ComplementNB, MultinomialNB
from sklearn.svm import LinearSVC


# Function to load training and testing data
def load_data():
    data_train = np.load("data/data_train.npy")  # Load training features from .npy file
    label_train = np.genfromtxt("data/label_train.csv", dtype=int, delimiter=",")[
        1:
    ]  # Load training labels from CSV, skipping the header
    data_test = np.load("data/data_test.npy")  # Load testing features from .npy file
    return data_train, label_train, data_test


# Function to get WordNet POS tags for lemmatization
def get_wordnet_pos(word):
    tag = pos_tag([word])[0][1][0].upper()  # Get the first letter of the POS tag
    tag_dict = {
        "J": wordnet.ADJ,  # Map "J" to adjective
        "N": wordnet.NOUN,  # Map "N" to noun
        "V": wordnet.VERB,  # Map "V" to verb
        "R": wordnet.ADV,  # Map "R" to adverb
    }
    return tag_dict.get(tag, wordnet.NOUN)  # Default to noun if no match


# Main function
def main():
    wnl = WordNetLemmatizer()  # Initialize lemmatizer
    vocab_maps = np.load("data/vocab_map.npy", allow_pickle=True)  # Load vocabulary map
    lemmatized_vocabs = [
        wnl.lemmatize(word, get_wordnet_pos(word)) for word in vocab_maps
    ]  # Lemmatize each word in the vocabulary map
    indices_to_remove = []  # List to store indices of words to remove

    index_map = defaultdict(list)  # Dictionary to map lemmatized words to indices
    for index, value in enumerate(lemmatized_vocabs):  # Iterate over lemmatized words
        if value in stopwords.words("english"):  # If the word is a stopword
            indices_to_remove.append(index)  # Mark the index for removal
        index_map[value].append(index)  # Map the word to its indices in the vocabulary
    data_train, label_train, data_test = load_data()  # Load training and testing data

    """
    # Code block to merge features with the same lemmatized form
    matching_indices = list(index_map.values())
    for indices in matching_indices:
        main_index = indices[0]
        for assoc_index in indices[1:]:
            data_train[:, main_index] += data_train[:, assoc_index]
            data_test[:, main_index] += data_test[:, assoc_index]
            indices_to_remove.append(assoc_index)
    """

    X_train = np.delete(
        data_train, indices_to_remove, 1
    )  # Remove columns of stopwords from training data
    X_test = np.delete(
        data_test, indices_to_remove, 1
    )  # Remove columns of stopwords from testing data

    # Split the data into training and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, label_train[:, 1], test_size=0.2, random_state=42
    )

    transformer = TfidfTransformer()  # Initialize TF-IDF transformer
    X_train_tf_idf = transformer.fit_transform(
        X_train
    ).toarray()  # Transform training data
    X_val_tf_idf = transformer.fit_transform(
        X_val
    ).toarray()  # Transform validation data
    X_test_tf_idf = transformer.fit_transform(
        X_test
    ).toarray()  # Transform testing data

    alpha_vals = np.arange(0.1, 2.1, 0.1)  # Alpha values to test for ComplementNB
    train_f1_scores = []  # List to store F1 scores for training data
    validation_f1_scores = []  # List to store F1 scores for validation data

    # Loop over alpha values and train ComplementNB model
    for alpha in alpha_vals:
        clf = ComplementNB(alpha=alpha)  # Initialize Complement Naive Bayes classifier
        clf.fit(X_train_tf_idf, y_train)  # Fit the model on training data
        y_val_pred = clf.predict(X_val_tf_idf)  # Predict on validation data
        y_train_pred = clf.predict(X_train_tf_idf)  # Predict on training data
        train_f1_score = f1_score(
            y_train, y_train_pred, average="macro"
        )  # Calculate F1 score for training
        validation_f1_score = f1_score(
            y_val, y_val_pred, average="macro"
        )  # Calculate F1 score for validation
        print(
            f"alpha: {alpha}, Train F1 Score: {train_f1_score:.3f}, Validation F1 Score: {validation_f1_score:.3f}"
        )  # Print F1 scores for the current alpha
        train_f1_scores.append(train_f1_score)  # Store training F1 score
        validation_f1_scores.append(validation_f1_score)  # Store validation F1 score

    # Plot F1 scores for different alpha values
    plt.figure(figsize=(8, 6))
    plt.plot(alpha_vals, train_f1_scores, label="Train F1 Score", marker="o")
    plt.plot(alpha_vals, validation_f1_scores, label="Validation F1 Score", marker="o")
    plt.title(
        "Complement Naive Bayes train vs validation F1 Score for Different Alpha Values"
    )
    plt.xlabel("alpha")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.grid(True)
    plt.savefig("cnb_f1_score.png")  # Save the plot as an image file
    plt.show()

    # Choose the alpha with the highest validation F1 score
    alpha_opt = alpha_vals[np.argmax(validation_f1_scores)]  # Optimal alpha
    clf = ComplementNB(alpha=alpha_opt)  # Initialize model with optimal alpha
    clf.fit(X_train_tf_idf, y_train)  # Train model on training data
    print(
        classification_report(y_val, clf.predict(X_val_tf_idf)), [0, 1]
    )  # Print classification report on validation data

    # Train the model on both training and validation data
    clf.fit(
        np.concatenate((X_train_tf_idf, X_val_tf_idf), axis=0),
        np.concatenate((y_train, y_val), axis=0),
    )

    y_pred = clf.predict(X_test_tf_idf)  # Predict on test data
    indices = np.arange(len(y_pred))  # Generate indices for the predictions

    # Write predictions to a CSV file
    with open("pred.csv", "w") as f:
        f.write("ID,label\n")
        for each_id, label in zip(indices, y_pred):
            line = f"{each_id},{label}\n"  # Write each prediction as ID,label
            f.write(line)


# Run the main function
if __name__ == "__main__":
    main()
