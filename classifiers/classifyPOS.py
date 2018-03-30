#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 23:39:36 2018

@author: jmkovachi
"""

from read_movie_reviews import read_Movies as movie_reader
import nltk
import random
from training import Data

"""def nltk_train(base_dir):
    mv = movie_reader()
    words = mv.tag_dir(base_dir)
    random.shuffle(words)
    train_set, test_set = words[0:(len(words)*3/4)], words[(len(words)*3/4):]
    print (words)
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    
    print(nltk.classify.accuracy(classifier, test_set))"""

def word_feats(words):
    return dict([(word, True) for word in words])


trial = [
  {
    "id": 23,
    "title": "Tesco says UK store closures put 2000 jobs at risk",
    "company": "Tesco",
    "sentiment": -0.9
  },
  {
    "id": 90,
    "title": "CRH's concrete bid for Holcim Lafarge assets",
    "company": "CRH",
    "sentiment": 0.3
  },
  {
    "id": 91,
    "title": "CRH's concrete bid for Holcim Lafarge assets",
    "company": "Holcim Lafarge",
    "sentiment": 0.3
  },
  {
    "id": 98,
    "title": "Reed Elsevier share price slides on underwhelming full-year results",
    "company": "Reed Elsevier",
    "sentiment": -0.9
  },
  {
    "id": 239,
    "title": "Kingfisher bid for Mr Bricolage runs into trouble",
    "company": "Kingfisher",
    "sentiment": -0.3
  },
  {
    "id": 240,
    "title": "Kingfisher bid for Mr Bricolage runs into trouble",
    "company": "Mr Bricolage",
    "sentiment": -0.3
  },
  {
    "id": 261,
    "title": "Glencore's annual results beat forecasts",
    "company": "Glencore",
    "sentiment": 0.9
  },
  {
    "id": 304,
    "title": "Diageo stays neutral on India boardroom turmoil",
    "company": "Diageo",
    "sentiment": -0.2
  },
  {
    "id": 675,
    "title": "Shell to Cut 6500 Jobs as Profit Drops",
    "company": "Shell",
    "sentiment": -1
  },
  {
    "id": 1210,
    "title": "Markets Shire up 2.5% and Baxalta up 6% on $32bn deal",
    "company": "Shire",
    "sentiment": 0.8
  },
{
    "id": 1211,
    "title": "Markets Shire up 2.5% and Baxalta up 6% on $32bn deal",
    "company": "Baxalta",
    "sentiment": 0.8
  },
  {
    "id": 1374,
    "title": "BP ends 27-year sponsorship of Tate as falling oil price takes toll",
    "company": "BP",
    "sentiment": -0.2
  },
  {
    "id": 1658,
    "title": "HSBC, Standard Chartered Lead Asia Bank Rout as U.K. Votes 'Out'",
    "company": "HSBC",
    "sentiment": -1
  },
  {
    "id": 1659,
    "title": "HSBC, Standard Chartered Lead Asia Bank Rout as U.K. Votes 'Out'",
    "company": "Standard Chartered",
    "sentiment": -1
  }
]

class Trainer:

    def __init__(self):
        self.classifier = None


    def nltk_train_semeval(self):
        d = Data()
        data = d.data
        print(data)
        words = [ (word_feats(nltk.word_tokenize(obj['title'])), 'positive') for obj in data if obj['sentiment'] > 0.25 ]
        words.extend([ (word_feats(nltk.word_tokenize(obj['title'])), 'negative') for obj in data if obj['sentiment'] < -0.25 ])
        #print(words)
        random.shuffle(words)
        train_set, test_set = words[0:int(len(words)*3/4)], words[int(len(words)*3/4)+1:]
        self.classifier = nltk.NaiveBayesClassifier.train(train_set)

        """print(classifier.classify(word_feats('Dow Industrials Rise, Extending Rally'.split(' '))))

        for t in trial:
            print(classifier.classify(word_feats(t['title'].split(' '))))
            print(t['sentiment'])
            print('\n')"""

    def classify(self, text):
        decision = self.classifier.classify(word_feats(nltk.word_tokenize(text)))
        return decision


nltk_train_semeval()
    
    
#nltk_train('/home/jmkovachi/sent-classifier/movie_reviews/txt_sentoken/')
        
        