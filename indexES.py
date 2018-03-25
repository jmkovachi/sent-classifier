import re
from elasticsearch import Elasticsearch
import time
import os
es = Elasticsearch()

class IndexES:

    def extract_header(text):
        """
        Extracts the header and text from a given article. 
        @param text: Article text to be extracted
        return: title, author, date, link to article, and body of text
        """
        search = re.search(r'--(.+?)--(.+?)--(.+?)--(.+?)Reuters\)\s-', text, flags=re.DOTALL)
        text = re.sub(r'--.+?--.+?--.+?--.+?Reuters\)\s-', '', text)
        title = search.group(1)
        author = search.group(2)
        date = search.group(3)
        link = search.group(4)
        return title, author, date, link, text


    def correct_date_format(string):
        """
        Gives a correct date format to be entered into the database. This date format is
        compatible with the Quandl API and can be passed directly as a parameter.
        @param string: Date string to be converted
        return: converted time

        Example input and output is below.

        Mon Jan 5, 2009 7:53pm EST

        2018-03-17T01h23m24

        """
        string = re.sub(r'\s(\d),', r' 0\1', string)
        string = re.sub(r'(\d):(\d\d)', r'0\1:\2', string)
        t = time.strptime(string, '%a %b %d %Y %H:%M%p %Z')
        t = time.strftime('%Y-%m-%d-%yT%Hh%Mm%S', t)
        return t

    def read_and_index():
        """
        Reads files and subsequently indexes them into an ElasticSearch database. Used for easily querying needed articles.
        """

        path = os.getcwd() 
        reuters_folders = os.listdir('/home/jmkovachi/Documents/jupyter_notebooks/reuters')[0:10]
        path += '/reuters'
        articles = []
        id = 1
        for folder in reuters_folders:
            article_files = os.listdir(path + '/' + folder)
            for file in article_files:
                with open(path + '/' + folder + '/' + file) as f:
                    raw_text = f.read()
                    title, author, date, link, text = extract_header(raw_text)
                    es.index(index='articles', doc_type='article', id=id, body={'title' : title.replace('\n',''), 'author' : author.replace('\n',''), 'date' : date.replace('\n',''), 'link' : link.replace('\n',''), 'text' : text.replace('\n','')})
                    id += 1

