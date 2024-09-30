'''
This file is for evaluating the quality of our RAG system using the Hairy Trumpet dataset.
'''

import ragnews
import json
import logging
import argparse
from sklearn.metrics import accuracy_score

# Setting up logging
logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    level=logging.INFO
)
logger = logging.getLogger()

class RAGEvaluator:
    # Specifying valid labels to predict with __init__ function
    def __init__(self, labels):
        self.valid_labels = labels
        self.db = ragnews.ArticleDB('ragnews.db')

    # Creating predictor part of the class
    def predict(self, masked_text):
        '''
        >>> model = RAGEvaluator()
        >>> model.predict('There is no mask token here')
        []
        >>> model.predict("The incumbent president, [MASK0], a member of the Democratic Party, initially ran for re-election and became the party's presumptive nominee, facing little opposition. However, [MASK0]'s performance in the June 2024 presidential debate intensified concerns about his age and health, and led to calls within his party for him to leave the race. Although initially adamant that he would remain in the race, [MASK0] ultimately withdrew on July 21 and endorsed Harris, who became the party's nominee on August 5. Harris selected Walz, the governor of Minnesota, as her running mate. [MASK0]'s withdrawal makes him the first eligible incumbent president since Lyndon B. Johnson in 1968 not to run for re-election, and the first to withdraw after securing enough delegates to win the nomination. Harris is the first nominee who did not participate in the primaries since Vice President Hubert Humphrey, also in 1968.")
        ['Biden']
        >>> model.predict("The incumbent president, Biden, a member of the Democratic Party, initially ran for re-election and became the party's presumptive nominee, facing little opposition. However, Biden's performance in the June 2024 presidential debate intensified concerns about his age and health, and led to calls within his party for him to leave the race. Although initially adamant that he would remain in the race, Biden ultimately withdrew on July 21 and endorsed [MASK0], who became the party's nominee on August 5. [MASK0] selected Walz, the governor of Minnesota, as her running mate. Biden's withdrawal makes him the first eligible incumbent president since Lyndon B. Johnson in 1968 not to run for re-election, and the first to withdraw after securing enough delegates to win the nomination. [MASK0] is the first nominee who did not participate in the primaries since Vice President Hubert Humphrey, also in 1968.")
        ['Harris']
        '''
        textprompt = f'''
This is a fancier question that is based on standard cloze style benchmarks. 
I am going to provide you a sentence, and that sentence will have a mask token inside of it that will look like [MASK0] or [MASK1] or [MASK2] and so on. 
Your job is to tell me what the value of that mask token was. 
The size of your output should just be a single word for each mask. If there is one mask, output one word. For example, if you believe the output should be 'Harris,' only output 'Harris,' not 'Harris, Harris, Harris.'
You should not provide any explanation or other extraneous words in the output, only provide a single word for each mask, with absolutely nothing else. 
You will get a prize if you respond with just the masked token(s). You will be punished if you output your thought process, a description of the information you're learning from, or any information that is not the masked tokens.
If there are multiple mask tokens, provide each token separately with whitespace in between. 
Valid values include: {', '.join(self.valid_labels)}

INPUT:[MASK0] is the current democratic presidential nominee
OUTPUT: 'Harris'

INPUT: [MASK0] is the democratic nominee and [MASK1] is the republican nominee
OUTPUT:'Harris, Trump'

INPUT: {masked_text}
OUTPUT: '''
        output = ragnews.rag(textprompt, self.db, keywords_text=masked_text)
        return output

#Evaluating rag function with hairy-trumpet data set
def main(data_file):
    labels = []  
    masked_texts = []

    # Extracting labels and masked texts from the hairy-trumpet dataset
    with open(data_file) as fin:
        for line in fin:
            dp = json.loads(line)
            masks = dp['masks']
            labels.extend(masks) 
            masked_texts.append(dp['masked_text'])

    logger.info(f'Extracted labels: {labels}')

    evaluator = RAGEvaluator(labels)

    # Predicting labels
    predicted_labels = []
    for masked_text in masked_texts:
        prediction = evaluator.predict(masked_text)
        if prediction: 
            predicted_labels.append(prediction)
        else:
            predicted_labels.append([])

    labels_flat = labels

    print("predicted labels = ", predicted_labels)
    print("labels =", labels_flat)

    # Calculating accuracy
    accuracy = accuracy_score(labels_flat, predicted_labels)
    logger.info(f'Accuracy: {accuracy:.2f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate RAG system using Hairy Trumpet dataset.')
    parser.add_argument('path', type=str, help='ragnews/hairy-trumpet')
    args = parser.parse_args()

    main(args.path)  # Passing args.path to the main function


