# Retrieval Augmented Generation (RAG) News Project ![](https://github.com/alexcnsf/RAGNews/blob/main/.github/workflows/tests/badge.svg)

This repository using RAG to equip GROQ's low-latency LLM API with up-to-date information regarding current politics. This allows it to answer questions, such as who is the Democractic Presidential Nominee non-RAG LLM's are unable to answer correctly. 

## Prereqs

- Before running please install all the libraries listed in the requirements.txt file.
- And input your own GROQ API key into your `.env` file in the main directory. You can create your own key from here [Groq API KEY](https://groq.com).

## Usage Steps

To use ragnews.py, follow these steps:

1. Ensure your .env file is configured correctly with your Groq API key and connect your .env by running the line:

```
$ export $(cat .env)
```

2. Run the ragnews.py script with the command:

```
python3 ragnews.py
```

3. The system will prompt you with:

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
```

