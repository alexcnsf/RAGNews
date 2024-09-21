import os
import sys
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from ragnews import rag, ArticleDB  # Import the correct functions and classes from ragnews

class RAGClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, valid_labels, db, model="llama3-8b-8192", num_keywords=10, num_articles=5, use_summary=True):
        """
        Initialize the classifier with valid labels and the article database.
        :param valid_labels: list of valid labels
        :param db: An instance of the ArticleDB class
        :param model: Model to use for the LLM (e.g., llama3-8b-8192)
        :param num_keywords: Number of keywords to extract for RAG search
        :param num_articles: Number of articles to include in the prompt
        :param use_summary: Whether to use summarized text for RAG predictions
        """
        self.valid_labels = valid_labels
        self.db = db
        self.model = model
        self.num_keywords = num_keywords
        self.num_articles = num_articles
        self.use_summary = use_summary

    def fit(self, X, y):
        """Dummy fit method to comply with scikit-learn interface."""
        return self

    def predict(self, X, question):
        """
        Use the rag function to predict the labels for the input data, using a user-provided question.
        :param X: Input data (list of text strings) for which to make predictions
        :param question: The question prompt provided by the user
        :return: Predicted labels
        """
        predictions = []
        for text in X:
            # Generate the RAG-based response for each text
            prediction = rag(question, self.db)
            
            # For this example, we're assuming the model provides a single label prediction based on the RAG output.
            # We will extract the predicted label from the response. You'll need to define how this works.
            # For now, we use a dummy placeholder 'label1' or 'label2' (modify based on real logic).
            
            if "[MASK0]" in text:  # Dummy condition based on the mask
                predictions.append('label1')  # You will want to replace this with actual label extraction logic
            else:
                predictions.append('label2')

        return predictions

def load_data(file_path):
    """Load the HairyTrumpet data from the specified file path."""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    data = {
        "text": [line.strip() for line in lines],  # Assuming each line is a text entry
        "label": ["label1" if i % 2 == 0 else "label2" for i in range(len(lines))]  # Dummy labels for now
    }
    
    return pd.DataFrame(data)

def main(datafile_path):
    # Load data
    data = load_data(datafile_path)

    labels = data['label'].unique()  # Extract unique valid labels
    X = data['text'].tolist()        # Input data
    y_true = data['label'].tolist()  # True labels

    # Initialize the ArticleDB
    db = ArticleDB()

    # Ask the user for the question prompt
    question = input("Please enter your question: ")

    # Initialize and fit the classifier with hyperparameters
    classifier = RAGClassifier(
        valid_labels=labels,
        db=db,
        model="llama3-8b-8192",       # Choose model
        num_keywords=10,              # Tweakable hyperparameter
        num_articles=5,               # Tweakable hyperparameter
        use_summary=True              # Choose between summaries or raw text
    )
    classifier.fit(X, y_true)  # Fit is not really used but kept for consistency

    # Make predictions using the user-provided question
    y_pred = classifier.predict(X, question)

    # Compute accuracy
    correct_predictions = sum(1 for true, pred in zip(y_true, y_pred) if true == pred)
    accuracy = correct_predictions / len(y_true)
    print(f"Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python evaluate.py <path_to_hairy_trumpet_datafile>")
        sys.exit(1)

    datafile_path = sys.argv[1]
    main(datafile_path)

