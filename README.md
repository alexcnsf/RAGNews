# RAG GROQ LLM for Politcial News ![](https://github.com/alexcnsf/RAGNews/workflows/tests/badge.svg)

This repository using RAG to equip GROQ's low-latency LLM API with up-to-date information regarding current politics. This allows it to answer questions, such as who is the Democractic Presidential Nominee non-RAG LLM's are unable to answer correctly. 

## Usage Notes

- **Requirements**: All requirements and dependencies are listed in the up-to-date requirements.txt file.
- **API Key** You have ot create your own [Groq API key](https://groq.com) and store it as GROQ_API_KEY in your `.env` file (e.g. `GROQ_API_KEY=your_key_here`) and you must initialize the enviromental variable using code: ` $ export $(cat .env) `

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
```

