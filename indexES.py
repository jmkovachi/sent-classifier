import re
from elasticsearch import Elasticsearch
import time
es = Elasticsearch()


def extract_header(text):
    search = re.search('--(.+?)--(.+?)--(.+?)--(.+?)Reuters\)\s-', text, flags=re.DOTALL)
    text = re.sub('--.+?--.+?--.+?--.+?Reuters\)\s-', '', text)
    title = search.group(1)
    author = search.group(2)
    date = search.group(3)
    link = search.group(4)
    return title, author, date, link, text

import os

path = os.getcwd() 

reuters_folders = os.listdir('/home/jmkovachi/Documents/jupyter_notebooks/reuters')[0:10]

path += '/reuters'

articles = []

"""

Mon Jan 5, 2009 7:53pm EST

2018-03-17T01h23m24

"""

t = time.strptime('Mon Jan 5, 2009 7:53pm'.replace(',',''), '%a %b %d %Y %H:%M%p %z')
print(t)
"""

id = 1
for folder in reuters_folders[:1]:
    article_files = os.listdir(path + '/' + folder)
    for file in article_files:
        with open(path + '/' + folder + '/' + file) as f:
            raw_text = f.read()
            title, author, date, link, text = extract_header(raw_text)
            es.index(index='articles', doc_type='article', id=id, body={'title' : title.replace('\n',''), 'author' : author.replace('\n',''), 'date' : date.replace('\n',''), 'link' : link.replace('\n',''), 'text' : text.replace('\n','')})
            id += 1

"""