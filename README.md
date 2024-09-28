# RAG Classifier Evaluation

This script runs a retrieval-augmented generation (RAG) classifier on the HairyTrumpet dataset using the Groq API. The goal is to achieve at least 70% accuracy on the specified data file.

## Running the Classifier

To run the classifier, use the following command:

```bash
python ragnews/evaluate.py /path/to/hairy-trumpet/data/wiki__page=2024_United_States_presidential_election,recursive_depth=0__dpsize=paragraph,transformations=[canonicalize, group, rmtitles, split]
```

### Results

Here is an example output on one of the articles:

```

</context>
<question>
The New York Times and Los Angeles Times editorial boards also declared [MASK0] unfit to lead, pointing to what former [MASK0] officials "have described as his systematic dishonesty, corruption, cruelty and incompetence." [MASK0] has also been criticized for his hiring decisions, and noted for his unusual criminal record.
</question>

model: llama-3.1-8b-instant
seed: None
temperature: 0.5
stop: </answer>
INFO:httpx:HTTP Request: POST https://api.groq.com/openai/v1/chat/completions "HTTP/1.1 200 OK"
INFO:root:run_llm result:
Based on the context provided, it seems that the article is discussing former President Donald Trump. The article mentions his critics and the New York Times and Los Angeles Times editorial boards declaring him unfit to lead.

The article also mentions that former officials have described Trump as having "systematic dishonesty, corruption, cruelty and incompetence." This is consistent with criticism that has been leveled against Trump throughout his presidency.

Therefore [MASK0] is Trump.

<answer>
Trump

Predicted labels: ['Trump']
Actual labels: ['Trump']
```

Where it correctly got the mask ['Trump']

And overall it scored an accuracy of:
```
Accuracy score: 74%:
```
