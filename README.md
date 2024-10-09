# RAG GROQ LLM for Politcial News ![](https://github.com/alexcnsf/RAGNews/workflows/tests/badge.svg)

This repository using RAG to equip GROQ's low-latency LLM API with up-to-date information regarding current politics. This allows it to answer questions, such as who is the Democractic Presidential Nominee non-RAG LLM's are unable to answer correctly. 

## Usage Notes

- **Requirements**: All requirements and dependencies are listed in the up-to-date requirements.txt file.
- **API Key** You have ot create your own [Groq API key](https://groq.com) and store it as GROQ_API_KEY in your `.env` file (e.g. `GROQ_API_KEY=your_key_here`) and you must initialize the enviromental variable using code: 
```
$ export $(cat .env)
``` 

To use ragnews.py, follow these steps:

1. Run the ragnews.py script with the command: 
``` 
python3 ragnews.py 
```

2. The system will prompt you with: 
```
 ragnews> 
```

4. And ask it anything you are wondering such as: 
``` 
ragnews> Who is the current Democractic Parties Presidential candidate? 
```

And it will give an output like this!

```
Based on the provided articles, the current Democratic Party presidential candidate is Kamala Harris, who officially became a candidate on Wednesday, August 21, 2024, accepting the nomination at a virtual Democratic National Convention in Wisconsin. Harris is the first woman of color to receive a major party's presidential nomination and has expressed her gratitude for the honor, stating: "For us, it's a privilege to be your nominees. This is a campaign founded on the power of the people, and together, we will pave a new path forward."
```


5. When you want to leave the program just use the command: 
```
control^ + C 


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

