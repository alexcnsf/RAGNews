import os
import sys
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from ragnews import rag, ArticleDB, summarize_text, translate_text, extract_keywords

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
        """
        Dummy fit method to comply with scikit-learn interface.
        This classifier does not actually train on the data.
        """
        return self

    def predict(self, X):
        """
        Use the rag function to predict the labels for the input data.
        :param X: Input data (list of text strings) for which to make predictions
        :return: Predicted labels
        """
        predictions = []
        for text in X:
            # Use extracted keywords and perform RAG
            keywords = extract_keywords(text, seed=42)[:self.num_keywords]
            articles = self.db.find_articles(' '.join(keywords))[:self.num_articles]
            
            # Use summaries, translations, or raw text for predictions
            articles_content = "\n\n".join([
                f"{article['title']}\n{article['en_summary'] if self.use_summary else article['text']}"
                for article in articles
            ])
            
            # Construct RAG prompt
            system = f"You are a professional journalist answering questions from provided articles."
            user = f"Answer the question using the provided articles: {text}\n\nArticles:\n{articles_content}"
            
            prediction = rag(user=user, system=system, model=self.model, db=self.db)
            predictions.append(prediction)
        return predictions

def load_data(file_path):
    """
    Load the HairyTrumpet data from the specified file path.
    """
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

    # Make predictions
    y_pred = classifier.predict(X)

    # Compute accuracy
    accuracy = sum(1 for true, pred in zip(y_true, y_pred) if true == pred) / len(y_true)
    print(f"Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python evaluate.py <path_to_hairy_trumpet_datafile>")
        sys.exit(1)

    datafile_path = sys.argv[1]
    main(datafile_path)

