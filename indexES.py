import re
from elasticsearch import Elasticsearch
import time
import os
import codecs
es = Elasticsearch()



def correct_date_format(string):
    """
    Gives a correct date format to be entered into the database. This date format is
    compatible with the Quandl API and can be passed directly as a parameter.
    @param string: Date string to be converted
    return: converted time

    Example input and output is below.

    Input: Mon Jan 5, 2009 7:53pm EST

    Return: 2009-01-05T07h53m00

    """
    print(string)
    string = re.sub(r'\s(\d),', r' 0\1,', string)
    string = string.replace(',','')
    string = re.sub(r'\s(\d):(\d\d)', r' 0\1:\2', string)
    t = time.strptime(string, '%a %b %d %Y %H:%M%p %Z')
    t = time.strftime('%Y-%m-%dT%Hh%Mm%S', t)
    return t


def extract_header(text):
    """
    Extracts the header and text from a given article. 
    @param text: Article text to be extracted
    return: title, author, date, link to article, and body of text
    """
    search = re.search('--(.+?)\n--(.+?)\n--(.+?)\n--(.+?)\n.+?Reuters\)\s-', text, flags=re.DOTALL)
    text = re.sub('^--.+?Reuters\)\s-', '', text, flags=re.DOTALL)
    title = search.group(1)
    author = re.sub('By\s', '', search.group(2))
    date = correct_date_format(codecs.getdecoder('unicode_escape')(search.group(3))[0].replace('\n','').strip())
    link = search.group(4)
    return title, author, date, link, text


def read_and_index():
    """
    Reads files and subsequently indexes them into an ElasticSearch database. Used for easily querying needed articles.
    """
    path = os.getcwd() 
    reuters_folders = os.listdir('/home/jmkovachi/Documents/jupyter_notebooks/reuters')
    path += '/reuters'
    id = 1
    error_count = 0
    for folder in reuters_folders:
        try:
            article_files = os.listdir(path + '/' + folder)
        except Exception as e:
            print(e)
            print(article_files)
            error_count += 1
            continue
        for file in article_files:
            try:
                with open(path + '/' + folder + '/' + file) as f:
                    raw_text = f.read()
                    try:
                        title, author, date, link, text = extract_header(raw_text)
                    except Exception as e:
                        print(text)
                        print(e)
                        error_count += 1
                        continue
                    es.index(index='articles', doc_type='article', id=id, body={'title' : codecs.getdecoder('unicode_escape')(title)[0].replace('\n','').strip(),
                                                                                'author' : author.replace('\n','').strip(), 
                                                                                'date' : date,
                                                                                'link' : link.replace('\n','').strip(),
                                                                                'text' : codecs.getdecoder('unicode_escape')(text)[0].replace('\n','').strip()})
                    id += 1
            except Exception as e:
                print(e)
                error_count += 1
 
    print(error_count)


if __name__ == "__main__":
    read_and_index()
