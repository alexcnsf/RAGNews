# RAG Classifier Evaluation

This script runs a retrieval-augmented generation (RAG) classifier on the HairyTrumpet dataset using the Groq API. The goal is to achieve at least 70% accuracy on the specified data file.

## Running the Classifier

To run the classifier, use the following command:

```bash
python ragnews/evaluate.py /path/to/hairy-trumpet/data/wiki__page=2024_United_States_presidential_election,recursive_depth=0__dpsize=paragraph,transformations=[canonicalize, group, rmtitles, split]

