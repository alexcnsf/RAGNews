import os
import sys
import json
from sklearn.base import BaseEstimator, ClassifierMixin
from ragnews import rag, ArticleDB  # Import the correct functions and classes from ragnews

class RAGClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, db):
        self.db = db

    def predict(self, X):
        """
        Predict the masks for the input data using the RAG model.
        """
        predictions = []
        for data_point in X:
            masked_text = data_point['masked_text']
            true_masks = data_point['masks']

            # Use RAG to predict the masks
            predicted_masks = rag(masked_text, self.db)

            predictions.append((predicted_masks, true_masks))
        
        return predictions

def load_data(file_path):
    """Load the JSONL data."""
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def compute_accuracy(predictions):
    """Calculate the percentage of correct mask predictions."""
    correct = 0
    total = len(predictions)

    for i, (pred, true) in enumerate(predictions):
        print(f"Data Point {i+1}:")
        print(f"Predicted: {pred}")
        print(f"True: {true}")
        if pred == true:
            correct += 1

    accuracy = correct / total * 100
    print(f"Accuracy: {accuracy:.2f}%")
    return accuracy

def main(datafile_path):
    # Load data
    data = load_data(datafile_path)

    # Initialize the ArticleDB
    db = ArticleDB()

    # Initialize the RAG classifier
    classifier = RAGClassifier(db=db)

    # Predict masks
    predictions = classifier.predict(data)

    # Compute and print accuracy
    accuracy = compute_accuracy(predictions)
    print(f"Accuracy: {accuracy:.2f}%")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python evaluate.py <path_to_datafile>")
        sys.exit(1)

    datafile_path = sys.argv[1]
    main(datafile_path)

