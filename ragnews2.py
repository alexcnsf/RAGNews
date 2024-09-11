#!/bin/python3

'''
Run an interactive QA session with the news articles using the Groq LLM API and retrieval augmented generation (RAG).

New articles can be added to the database with the --add_url parameter,
and the path to the database can be changed with the --db parameter.
'''

from bs4 import BeautifulSoup
from urllib.parse import urlparse
import datetime
import logging
import re
import requests
import sqlite3
import random
import spacy

import groq
import metahtml

from groq import Groq
import os


################################################################################
# LLM functions
################################################################################

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)


def run_llm(system, user, model='llama3-8b-8192', seed=None):
    '''
    This is a helper function for all the uses of LLMs in this file.
    '''
    chat_completion = client.chat.completions.create(
        messages=[
            {
                'role': 'system',
                'content': system,
            },
            {
                "role": "user",
                "content": user,
            }

        ],
        model=model,
        seed=seed,
    )
    return chat_completion.choices[0].message.content


def summarize_text(text, seed=None):
    system = 'Summarize the input text below.  Limit the summary to 1 paragraph.  Use an advanced reading level similar to the input text, and ensure that all people, places, and other proper and dates nouns are included in the summary.  The summary should be in English.'
    return run_llm(system, text, seed=seed)


def translate_text(text):
    system = 'You are a professional translator working for the United Nations.  The following document is an important news article that needs to be translated into English.  Provide a professional translation.'
    return run_llm(system, text)


def extract_keywords(text, seed=None):
    system_prompt = '''You are an AI assistant tasked with extracting keywords from a given text. Your goal is to generate a list of all relevant and related words that capture the main ideas, topics, and concepts from the text. In addition to the core ideas, include any related words that provide additional context or depth.

Your output should consist of as many relevant and related keywords as possible. Focus on identifying both the key concepts and any closely connected ideas, actions, entities, or themes. Related terms are important, so include any words that are contextually important to the main topics of the text.

The output should be a space-separated list of words with no punctuation, no formatting, and no additional commentary. Your task is to list as many relevant words as possible that reflect both the direct content of the text and any associated or related ideas. There is no limit to the number of words—include every relevant keyword and related concept you can think of.

Do not include common filler words like “the,” “is,” “and,” “of,” or any other similar words that do not add value. Focus on the important words that provide meaningful context and help summarize the text. You can include nouns, verbs, proper nouns, adjectives, and any other terms that are useful for understanding the main ideas.

If the text contains complex or broad topics, include related concepts to provide a more comprehensive understanding. There is no need to limit yourself to just the most obvious keywords—add all important and related words that could help explain or expand upon the main ideas.

Your only output should be a space-separated list of words, without any punctuation or extra formatting. Focus on providing as many relevant and related words as possible. The more words that accurately reflect the content of the text, the better.

Compound concepts like "climate change" or "border control" should be included, but they should be separated by a space, not combined with punctuation. The output must contain only words, and all relevant and related terms should be included to capture the full context of the text. You are an AI assistant tasked with extracting keywords from a given text. Your goal is to generate a list of all relevant and related words that capture the main ideas, topics, and concepts from the text. In addition to the core ideas, include any related words that provide additional context or depth.

Your output must consist only of space-separated keywords. **Do not include any explanations, notes, or commentary of any kind.** The only acceptable output is a list of words separated by spaces. There should be no punctuation, labels, numbers, or additional text. 

The task is to provide as many relevant and related words as possible that reflect the main topics, ideas, and any closely connected concepts. Include key terms, actions, people, places, and themes that are important to understanding the text. There is no limit to how many words you can include.

Do not include any filler words such as "the," "is," "and," "of," or any other similar words that do not add value. Focus on meaningful words that summarize the key content and any related ideas. You can include nouns, verbs, adjectives, proper nouns, and compound concepts, but compound terms like "border control" should be separated by a space.

Remember: **Do not include any notes or explanations**. Only output a space-separated list of relevant words.'''

    # Define the user prompt as the input text
    user_prompt = f"Extract keywords from the following text: {text}"

    # Call the run_llm function to get the keywords
    keywords = run_llm(system_prompt, user_prompt, seed=seed)

    # Return the result from the LLM (assuming it's already a space-separated string of keywords)
    return keywords

# Example usage and test cases
if __name__ == "__main__":
    print(extract_keywords('What are good places to fish in Southern California?', seed=0))
    print(extract_keywords('Who is the current democratic presidential nominee?', seed=0))
    print(extract_keywords('What is the policy position of Trump related to illegal Mexican immigrants?', seed=0))
