import re
import csv
from elasticsearch import Elasticsearch
import datetime
import os
import codecs
from pymongo import MongoClient
import pymongo
es = Elasticsearch()
#client = MongoClient()
client = MongoClient("mongodb://127.0.0.1:27018")
db = client['primer']

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
    
    time = datetime.datetime.strptime(string, '%a %b %d %Y %H:%M%p %Z')
    format_time = datetime.datetime.strftime(time, '%Y-%m-%dT%Hh%Mm%S')
    return time, format_time



def extract_header(text):
    """
    Extracts the header and text from a given article. 
    @param text: Article text to be extracted
    return: title, author, date, formatted time, link to article, and body of text
    """
    search = re.search(r'--(.+?)\n--(.+?)\n--(.+?)\n--(.+?)\n.+?Reuters\)\s-', text, flags=re.DOTALL)
    text = re.sub(r'^--.+?Reuters\)\s-', '', text, flags=re.DOTALL)
    title = search.group(1)
    author = re.sub(r'By\s', '', search.group(2))
    date, formatted_time = correct_date_format(codecs.getdecoder('unicode_escape')(search.group(3))[0].replace('\n','').strip())
    link = search.group(4)
    return title, author, date, formatted_time, link, text


def read_and_index():
    """
    Reads files and subsequently indexes them into an ElasticSearch database. Used for easily querying needed articles.
    """
    db.articles.drop()
    path = '/home/jmkovachi/Documents/jupyter_notebooks'
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
                        title, author, date, formatted_time, link, text = extract_header(raw_text)
                    except Exception as e:
                        #print(text)
                        print(e)
                        error_count += 1
                        continue
                    db.articles.insert_one({'title' : codecs.getdecoder('unicode_escape')(title)[0].replace('\n','').strip(),
                                                                                'author' : author.replace('\n','').strip(), 
                                                                                'date' : date,
                                                                                'time_string' : formatted_time,
                                                                                'link' : link.replace('\n','').strip(),
                                                                                'text' : codecs.getdecoder('unicode_escape')(text)[0].replace('\n','').strip()})
                    id += 1
            except Exception as e:
                print(e)
                error_count += 1
 
    print(error_count)

def read_company_list():
    with open('WIKI-datasets-codes.csv', 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',', quotechar='|')
        for row in reader:
            if 'Untitled' in row[1]:
                continue
            abbrev = row[0]
            title = row[1]
            abbrev = re.sub(r'WIKI/', '', abbrev, flags=re.DOTALL)
            title = re.sub(r'\sPrices.*', '', title[1:], flags=re.DOTALL)
            print('{}, {}'.format(abbrev, title))
            db.companies.insert_one({ 'code' : abbrev, 'title' : title})

if __name__ == "__main__":
    #db.articles.create_index( [( 'author', pymongo.TEXT),  ('title', pymongo.TEXT), ('time_string', pymongo.TEXT)] )
    db.companies.create_index( [('code', pymongo.TEXT), ('title', pymongo.TEXT)] )
    read_company_list()
    #read_and_index()
