import os
import sys
import pandas as pd
from ragnews import rag, ArticleDB  # Import the correct functions and classes from ragnews

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

    X = data['text'].tolist()  # Input data

    # Initialize the ArticleDB
    db = ArticleDB()

    # Ask the user for the question prompt
    question = input("Please enter your question: ")

    # Process each text entry using RAG
    for text in X:
        print(f"Processing entry: {text[:50]}...")  # Print the first 50 characters of the text for reference
        
        # Use RAG to generate a response to the user's question using the articles from the DB
        response = rag(question, db)
        
        # Print the response from the RAG model
        print(f"Response:\n{response}\n")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python evaluate.py <path_to_hairy_trumpet_datafile>")
        sys.exit(1)

    datafile_path = sys.argv[1]
    main(datafile_path)

