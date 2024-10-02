# Retrieval Augmented Generation (RAG) News Project ![](https://github.com/alexcnsf/RAGNews/workflows/tests/badge.svg)

# RAG Classifier Evaluation

This script runs a retrieval-augmented generation (RAG) classifier on the HairyTrumpet dataset using the Groq API. The goal is to achieve at least 70% accuracy on the specified data file.

## Running the Classifier

To run the classifier, use the following command:

```bash
$ python3 ragnews/evaluate.py /path/to/hairy-trumpet/data/wiki__page=2024_United_States_presidential_election,recursive_depth=0__dpsize=paragraph,transformations=[canonicalize, group, rmtitles, split]
```

### Results

Here is an example result for this command: 

```
$ python3 -m ragnews.evaluate "hairy-trumpet/data/wiki__page=2024_United_States_presidential_election,recursive_depth=0__dpsize=paragraph,transformations=[canonicalize, group, rmtitles, split]"
2024-09-30 02:01:33 INFO     Extracted labels: ['Biden', 'Harris', 'Walz', 'Biden', 'Trump', 'Vance', 'Trump', 'Trump', 'Biden', 'Trump', 'Trump', 'Trump', 'Biden', 'Trump', 'Trump', 'Biden', 'Biden', 'Trump', 'Trump', 'Biden', 'Harris', 'Trump', 'Biden', 'Trump', 'Trump', 'Trump', 'Trump', 'Trump', 'Trump', 'Biden', 'Biden', 'Trump', 'Trump', 'Trump', 'Trump', 'Biden', 'Trump', 'Biden', 'Trump', 'Harris', 'Trump', 'Biden', 'Trump', 'Harris', 'Trump', 'Harris', 'Harris', 'Trump', 'Trump', 'Biden', 'Harris', 'Trump', 'Trump', 'Trump', 'Trump', 'Harris', 'Biden', 'Trump', 'Trump', 'Biden', 'Trump', 'Trump', 'Harris', 'Biden', 'Harris', 'Harris', 'Biden', 'Harris', 'Trump', 'Harris', 'Trump', 'Biden', 'Trump', 'Harris', 'Harris', 'Biden', 'Trump', 'Biden', 'Trump', 'Biden', 'Harris', 'Biden', 'Trump', 'Harris', 'Trump', 'Harris', 'Trump', 'Biden', 'Biden', 'Harris', 'Biden', 'Biden', 'Biden', 'Biden', 'Trump', 'Harris', 'Biden', 'Walz', 'Harris', 'Walz', 'Trump', 'Biden', 'Trump', 'Trump', 'Trump', 'Trump', 'Trump', 'Trump', 'Trump', 'Trump', 'Vance', 'Trump', 'Oliver', 'Stein', 'Ware', 'Trump', 'Biden', 'Trump', 'Harris', 'Trump', 'Biden', 'Biden', 'Trump', 'Harris', 'Trump', 'Vance', 'Walz']

predicted labels =  ['Biden', 'Harris', 'Walz', 'Biden, Trump', 'Trump', 'Vance', 'Trump', 'Trump', 'Biden', 'Trump', 'Trump', 'Trump', 'Biden', 'Trump', 'Trump', 'Biden', 'Biden', 'Trump', 'Trump', 'Biden', 'Harris', 'Trump', 'Biden', 'Trump', 'Trump, Trump', 'Trump', 'Trump', 'Trump', 'Trump, Trump', 'Biden', 'Biden', 'Biden, Biden, Biden', 'Trump, Trump, Trump', 'Trump', 'Biden', 'Biden', 'Trump', 'Biden', 'Trump', 'Harris', 'Trump', 'Biden', 'Trump', 'Biden', 'Trump', 'Biden', 'Harris', 'Trump', 'Trump', 'Trump', 'Harris', 'Trump', 'Trump', 'Trump', 'Trump', 'Harris', 'Biden', 'Trump', 'Trump', 'Biden', 'Trump', 'Trump', 'Harris', 'Biden', 'Harris', 'Harris', 'Biden', 'Harris', 'Trump', 'Biden', 'Trump', 'Biden', 'Trump', 'Harris', 'Harris', 'Biden', 'Trump', 'Biden', 'Trump', 'Biden', 'Biden', 'Biden', 'Trump', 'Harris', 'Trump', 'Harris', 'Trump', 'Biden', 'Biden, Biden', 'Harris', 'Biden', 'Biden', 'Harris', 'Biden, Biden', 'Trump', 'Harris', 'Harris', 'Harris', 'Harris', 'Biden', 'Trump', 'Biden', 'Trump', 'Trump', 'Trump', 'Trump', 'Trump, Trump, Trump', 'Trump', 'Biden', 'Trump', 'Vance', 'Trump', 'Ware', 'Trump', 'Ware', 'Trump', 'Biden', 'Trump', 'Biden', 'Trump', 'Biden', 'Biden', 'Trump', 'Biden', 'Trump', 'Harris', 'Harris']
labels = ['Biden', 'Harris', 'Walz', 'Biden', 'Trump', 'Vance', 'Trump', 'Trump', 'Biden', 'Trump', 'Trump', 'Trump', 'Biden', 'Trump', 'Trump', 'Biden', 'Biden', 'Trump', 'Trump', 'Biden', 'Harris', 'Trump', 'Biden', 'Trump', 'Trump', 'Trump', 'Trump', 'Trump', 'Trump', 'Biden', 'Biden', 'Trump', 'Trump', 'Trump', 'Trump', 'Biden', 'Trump', 'Biden', 'Trump', 'Harris', 'Trump', 'Biden', 'Trump', 'Harris', 'Trump', 'Harris', 'Harris', 'Trump', 'Trump', 'Biden', 'Harris', 'Trump', 'Trump', 'Trump', 'Trump', 'Harris', 'Biden', 'Trump', 'Trump', 'Biden', 'Trump', 'Trump', 'Harris', 'Biden', 'Harris', 'Harris', 'Biden', 'Harris', 'Trump', 'Harris', 'Trump', 'Biden', 'Trump', 'Harris', 'Harris', 'Biden', 'Trump', 'Biden', 'Trump', 'Biden', 'Harris', 'Biden', 'Trump', 'Harris', 'Trump', 'Harris', 'Trump', 'Biden', 'Biden', 'Harris', 'Biden', 'Biden', 'Biden', 'Biden', 'Trump', 'Harris', 'Biden', 'Walz', 'Harris', 'Walz', 'Trump', 'Biden', 'Trump', 'Trump', 'Trump', 'Trump', 'Trump', 'Trump', 'Trump', 'Trump', 'Vance', 'Trump', 'Oliver', 'Stein', 'Ware', 'Trump', 'Biden', 'Trump', 'Harris', 'Trump', 'Biden', 'Biden', 'Trump', 'Harris', 'Trump', 'Vance', 'Walz']

2024-09-30 02:10:40 INFO     Accuracy: 0.80
```

This excitingly has exceeded on our goal of 70% accuracy!!

