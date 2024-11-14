# kaggle-competition
# Text Classification with Complement Naive Bayes

This project implements a text classification pipeline using a Complement Naive Bayes model. The pipeline includes preprocessing steps like lemmatization and stopword removal, followed by feature extraction using TF-IDF, and model training. The model is evaluated using F1 scores across different alpha values.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Dataset](#dataset)
- [Running the Script](#running-the-script)
- [Output](#output)
- [How the Code Works](#how-the-code-works)

## Prerequisites

Before running the script, ensure you have the following installed:

- Python 3.x
- Pip (Python package manager)

## Installation

1. **Clone the Repository**

   Start by cloning the repository or downloading the script to your local machine:

   ```bash
   git clone https://github.com/sokolat/kaggle-competition.git
   cd kaggle-competition

2. **Create a Virtual Environment (optional but recommended)
   
   It is recommended to use a virtual environment to manage dependencies:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate

4. **Install Required Packages
   
   Run the following command to install all the required dependencies:

   ```bash
   pip install numpy matplotlib nltk scikit-learn

5. **Download NLTK Data

   The script requires some NLTK resources for text processing. Download the necessary datasets:

   ```bash
   python -m nltk.downloader stopwords wordnet

## Dataset

Make sure you have the necessary dataset files in the data/ directory:

data_train.npy: A .npy file containing the training features.
label_train.csv: A .csv file containing the labels for the training data.
data_test.npy: A .npy file containing the test features.
vocab_map.npy: A .npy file containing the vocabulary map.

You should organize your data directory as follows:

      ```bash
      data/
        ├── data_train.npy
        ├── label_train.csv
        ├── data_test.npy
        └── vocab_map.npy

## Running the Script
To run the text classification script:

1. **Open a terminal or command prompt and navigate to the directory containing the script:

   ```bash
   cd path/to/your/script

2. **Execute the Python script:
   ```bash
   python text_classification.py

## Output

Once the script is run successfully, it will produce the following outputs:

- F1 Score Plot: The script generates a plot (cnb_f1_score.png) showing F1 scores for both the training and validation datasets across different alpha values.

- Predictions CSV: A pred.csv file is generated containing predictions on the test data. The file format is as follows:
