#!/bin/python3

'''
Run an interactive QA session with the news articles using the Groq LLM API and retrieval augmented generation (RAG).

New articles can be added to the database with the --add_url parameter,
and the path to the database can be changed with the --db parameter.
'''

from urllib.parse import urlparse
import datetime
import logging
import re
import sqlite3

from groq import Groq
import groq
import os

from bs4 import BeautifulSoup
import requests
#import ragnewsmetahtml


################################################################################
# LLM functions
################################################################################

client = Groq(
    api_key=os.environ.get("GROQ_API_KEY"),
)


def run_llm(system, user, model='llama-3.1-70b-versatile', seed=None):
    '''
    This is a helper function for all the uses of LLMs in this file.
    '''
    chat_completion = client.chat.completions.create(
        temperature = .5,
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
        seed=seed if seed is not None else None
        #seed allows for deterministic results
    )
    return chat_completion.choices[0].message.content


def summarize_text(text, seed=None):
    system = 'Summarize the input text below.  Limit the summary to 1 paragraph.  Use an advanced reading level similar to the input text, and ensure that all people, places, and other proper and dates nouns are included in the summary.  The summary should be in English.'
    return run_llm(system, text, seed=seed)


def translate_text(text):
    system = 'You are a professional translator working for the United Nations.  The following document is an important news article that needs to be translated into English.  Provide a professional translation.'
    return run_llm(system, text)


def extract_keywords(text, seed=None):
    r'''
    This is a helper function for RAG.
    Given an input text,
    this function extracts the keywords that will be used to perform the search for articles that will be used in RAG.

    >>> extract_keywords('Who is the current democratic presidential nominee?', seed=0)
    'Joe candidate nominee presidential Democrat election primary TBD voting politics'
    >>> extract_keywords('What is the policy position of Trump related to illegal Mexican immigrants?', seed=0)
    'Trump Mexican immigrants policy position illegal border control deportation walls'

    Note that the examples above are passing in a seed value for deterministic results.
    In production, you probably do not want to specify the seed.
    '''

    system = '''
    You are an advanced assistant specializing in text analysis. 
    Your task is to extract the most important and relevant keywords from the following text.  
    The keywords should capture the core topics and entities mentioned in the text.
    Responses should come in the format of 'keyword keyword keyword keyword'. 
    Output exactly 10 keywords. 
    Output only keywords, nothing more. 
    Do not prefice the keywords with "here are the keywords," and do not explain how the keywords were found or what their sources are.
    '''
  
    return run_llm(system, text, seed=seed)

    # FIXME:
    # Implement this function.
    # It's okay if you don't get the exact same keywords as me.
    # You probably certainly won't because you probably won't come up with the exact same prompt as me.
    # To make the test cases above pass,
    # you'll have to modify them to be what the output of your prompt provides.


################################################################################
# helper functions
################################################################################

def _logsql(sql):
    rex = re.compile(r'\s+')
    sql_dewhite = rex.sub(' ', sql).strip()
    logging.debug(f'SQL: {sql_dewhite}')


def _catch_errors(func):
    '''
    This function is intended to be used as a decorator.
    It traps whatever errors the input function raises and logs the errors.
    We use this decorator on the add_urls method below to ensure that a webcrawl continues even if there are errors.
    '''
    def inner_function(*args, **kwargs):
        try:
            func(*args, **kwargs)
        except Exception as e:
            logging.error(str(e))
    return inner_function


################################################################################
# rag
################################################################################


def rag(text, db, keywords_text=None):
    '''
    This function uses retrieval augmented generation (RAG) to generate an LLM response to the input text.
    The db argument should be an instance of the `ArticleDB` class that contains the relevant documents to use.
    '''
    if keywords_text is None:
          keywords_text = text
    #1. extract keywords from the text

    keywords = extract_keywords(keywords_text)

    #2. use key words to find articles related to the text
    articles = db.find_articles(keywords)

    articles_str = "\n".join([f"{article['title']} - {article['url']}" for article in articles])    
    # 3. Construct a new user prompt that includes all of the articles and the original text.
    user = f'''
    Here is the original text:
    {text}
    Here are relevant articles:
    {articles_str}
    Based on the original text and the relevant articles, please generate a detailed, accurate, and informative response to the user query.
    '''

    # Step 4: Define my own system prompt
    system = '''
    You are an advanced assistant specializing in analyzing news articles and providing insights on current events.
    Your task is to generate an accurate response to user queries based on the context provided by the articles.
    Your response should include no extraneous words and should not explain your thinking. 
    You will get a prize if you output exactly what is being asked for, and you will lose points if you output anything other than what is being asked for. 
    '''

    #logging.info('rag.SYSTEM: ' + system)
    #logging.info('rag.USER: ' + user)

    # Step 5: Pass the new prompts to the LLM and return the result
    response = run_llm(system, user)
    #response = " ".join(response.split("Keywords:**\n\n* ")[-1].split("\n* "))
    return response

    # FIXME:
    # Implement this function.
    # Recall that your RAG system should:
    # 1. Extract keywords from the text.
    # 2. Use those keywords to find articles related to the text.
    # 3. Construct a new user prompt that includes all of the articles and the original text.
    # 4. Pass the new prompt to the LLM and return the result.
    #
    # HINT:
    # You will also have to write your own system prompt to use with the LLM.
    # I needed a fairly long system prompt (about 15 lines) in order to get good results.
    # You can start with a basic system prompt right away just to check if things are working,
    # but don't spend a lot of time on the system prompt until you're sure everything else is working.
    # Then, you can iteratively add more commands into the system prompt to correct "bad" behavior you see in your program's output.


class ArticleDB:
    '''
    This class represents a database of news articles.
    It is backed by sqlite3 and designed to have no external dependencies and be easy to understand.

    The following example shows how to add urls to the database.

    #>>> db = ArticleDB() 
    #>>> len(db)
    0
    #>>> db.add_url(ArticleDB._TESTURLS[0])
    #>>> len(db)
    1

    Once articles have been added,
    we can search through those articles to find articles about only certain topics.

    #>>> articles = db.find_articles('Economía')

    The output is a list of articles that match the search query.
    Each article is represented by a dictionary with a number of fields about the article.

    #>>> articles[0]['title']
    'La creación de empleo defrauda en Estados Unidos en agosto y aviva el temor a una recesión | Economía | EL PAÍS'
    #>>> articles[0].keys()
    ['rowid', 'rank', 'title', 'publish_date', 'hostname', 'url', 'staleness', 'timebias', 'en_summary', 'text']
    '''

    _TESTURLS = [
        'https://elpais.com/economia/2024-09-06/la-creacion-de-empleo-defrauda-en-estados-unidos-en-agosto-y-aviva-el-fantasma-de-la-recesion.html',
        'https://www.cnn.com/2024/09/06/politics/american-push-israel-hamas-deal-analysis/index.html',
    ]

    def __init__(self, filename=':memory:'):
        self.db = sqlite3.connect(filename)
        self.db.row_factory = sqlite3.Row
        self.logger = logging
        self._create_schema()

    def _create_schema(self):
        '''
        Create the DB schema if it doesn't already exist.
        '''
        try:
            sql = '''
            CREATE VIRTUAL TABLE articles
            USING FTS5 (
                title,
                text,
                hostname,
                url,
                publish_date,
                crawl_date,
                lang,
                en_translation,
                en_summary
                );
            '''
            self.db.execute(sql)
            self.db.commit()
        except sqlite3.OperationalError:
            self.logger.debug('CREATE TABLE failed')

    def find_articles(self, query, limit=10, timebias_alpha=1):
        '''
        Return a list of articles in the database that match the specified query.

        Lowering the value of the timebias_alpha parameter will result in the time becoming more influential.
        The final ranking is computed by the FTS5 rank * timebias_alpha / (days since article publication + timebias_alpha).
        '''
        formatted_query = f'"{query}"'
        sql = '''
        SELECT url, title, en_summary
        FROM articles
        WHERE articles MATCH ?
        ORDER BY rank
        LIMIT ?;
        '''
        #print(sql)

        _logsql(sql)
        cursor = self.db.cursor()
        cursor.execute(sql, (formatted_query, limit))
        rows = cursor.fetchall()

        articles = []
        for row in rows:
            publish_date = datetime.datetime.fromisoformat(row['publish_date'])
            age_days = (datetime.datetime.now() - publish_date).days
            time_bias = timebias_alpha / (age_days + timebias_alpha)
            final_rank = row['rank'] * time_bias

            article = {
                'url': row['url'],
                'title': row['title'],
                'en_summary': row['en_summary'],
            }
            articles.append(article)

        articles.sort(key=lambda x: x['final_rank'], reverse=True)

        return articles


    @_catch_errors
    def add_url(self, url, recursive_depth=0, allow_dupes=False):
        '''
        Download the url, extract various metainformation, and add the metainformation into the db.
        '''
        logging.info(f'add_url {url}')

        if not allow_dupes:
            logging.debug(f'checking for url in database')
            sql = '''
            SELECT count(*) FROM articles WHERE url=?;
            '''
            _logsql(sql)
            cursor = self.db.cursor()
            cursor.execute(sql, [url])
            row = cursor.fetchone()
            is_dupe = row[0] > 0
            if is_dupe:
                logging.debug(f'duplicate detected, skipping!')
                return

        logging.debug(f'downloading url')
        try:
            response = requests.get(url)
        except requests.exceptions.MissingSchema:
            url = 'https://' + url
            response = requests.get(url)
        parsed_uri = urlparse(url)
        hostname = parsed_uri.netloc

        logging.debug(f'extracting information')
        #parsed = ragnewsmetahtml.parse(response.text, url)
        #info = ragnewsmetahtml.simplify_meta(parsed)

        if info['type'] != 'article' or len(info['content']['text']) < 100:
            logging.debug(f'not an article... skipping')
            en_translation = None
            en_summary = None
            info['title'] = None
            info['content'] = {'text': None}
            info['timestamp.published'] = {'lo': None}
            info['language'] = None
        else:
            logging.debug('summarizing')
            if not info['language'].startswith('en'):
                en_translation = translate_text(info['content']['text'])
            else:
                en_translation = None
            en_summary = summarize_text(info['content']['text'])

        logging.debug('inserting into database')
        sql = '''
        INSERT INTO articles(title, text, hostname, url, publish_date, crawl_date, lang, en_translation, en_summary)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
        '''
        _logsql(sql)
        cursor = self.db.cursor()
        cursor.execute(sql, [
            info['title'],
            info['content']['text'],
            hostname,
            url,
            info['timestamp.published']['lo'],
            datetime.datetime.now().isoformat(),
            info['language'],
            en_translation,
            en_summary,
        ])
        self.db.commit()

        logging.debug('recursively adding more links')
        if recursive_depth > 0:
            for link in info['links.all']:
                url2 = link['href']
                parsed_uri2 = urlparse(url2)
                hostname2 = parsed_uri2.netloc
                if hostname in hostname2 or hostname2 in hostname:
                    self.add_url(url2, recursive_depth - 1)

    def __len__(self):
        sql = '''
        SELECT count(*)
        FROM articles
        WHERE text IS NOT NULL;
        '''
        _logsql(sql)
        cursor = self.db.cursor()
        cursor.execute(sql)
        row = cursor.fetchone()
        return row[0]


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--loglevel', default='warning')
    parser.add_argument('--db', default='ragnews.db')
    parser.add_argument('--recursive_depth', default=0, type=int)
    parser.add_argument('--add_url', help='If this parameter is added, then the program will not provide an interactive QA session with the database.  Instead, the provided url will be downloaded and added to the database.')
    args = parser.parse_args()

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
        level=args.loglevel.upper(),
        )

    db = ArticleDB(args.db)
    #stores all of our articles

    if args.add_url:
        db.add_url(args.add_url, recursive_depth=args.recursive_depth, allow_dupes=True)

    else:
        import readline
        while True:
            text = input('ragnews> ')
            if len(text.strip()) > 0:
                output = rag(text, db)
                print(output)
