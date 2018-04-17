#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 23:39:36 2018

@author: jmkovachi
"""

#from classifiers.read_movie_reviews import read_Movies as movie_reader
import nltk
import random
import traceback
from training import Data
from pymongo import MongoClient
import datetime
from datetime import timedelta
import QuandlWrapper
import Reuters_PMI
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
import numpy as np
client = MongoClient("mongodb://127.0.0.1:27018")
db = client['primer']

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

  def __init__(self, year):
    self.classifier = None
    self.year = year

class NB_Trainer(Trainer):

    def __init__(self, year):
        self.classifier = None
        Trainer.__init__(self, year)


    def nltk_train_semeval(self):
        """
        This function trains the semeval data on the NLTK naivebayes classifier.
        """
        d = Data()
        data = d.data
        #

        # print(data)
        words = [ (word_feats(nltk.word_tokenize(obj['title'])), 'positive') for obj in data if obj['sentiment'] > 0.25 ]
        words.extend([ (word_feats(nltk.word_tokenize(obj['title'])), 'negative') for obj in data if obj['sentiment'] < -0.25 ])
        #print(words)
        random.shuffle(words)
        train_set = words
        self.classifier = nltk.NaiveBayesClassifier.train(train_set)

        """print(classifier.classify(word_feats('Dow Industrials Rise, Extending Rally'.split(' '))))

        for t in trial:
            print(classifier.classify(word_feats(t['title'].split(' '))))
            print(t['sentiment'])
            print('\n')"""

    def classify(self, text):
        decision = self.classifier.classify(word_feats(nltk.word_tokenize(text)))
        return decision

class SVM_Trainer(Trainer):

  def __init__(self, year):
    self.classifier = None
    Trainer.__init__(self, year)

  def train(self, train_titles=False):
    """
    Method to train the SDG classifier.
    @param train_titles: boolean to indicate whether titles or bodies of text should be trained.
    return: test set of articles that we will use for training.
    """
    articles = []
    decisions = []
    count_articles = 0
    count = 0
    train_set = []
    test_set = []
    for article in db.articles.find():
      if count_articles % 4 == 0:
        test_set.append(article)
      else:
        train_set.append(article)
      count_articles += 1
    for article in train_set:
      try:
        orgs = Reuters_PMI.find_incident_orgs(article['title'])
        if orgs == []:
          continue
        prices = QuandlWrapper.query_org_prices(orgs[0], QuandlWrapper.convert_dates([article['time_string'], QuandlWrapper.add_week(article['time_string'])]))
      except Exception as e:
        print(traceback.format_exc())
        continue
      articles.append(article['text' if not train_titles else 'title'])
      if prices['close'] > prices['open']:
        decisions.append('positive')
      else:
        decisions.append('negative')
      count += 1
      if count == 3000:
        break

      print(decisions[len(decisions)-1])

    print(len(decisions))

    decisions = np.array(decisions)
    self.count_vec = CountVectorizer()

    X_train_counts = self.count_vec.fit_transform(articles)

    
    self.tfidf_transformer = TfidfTransformer()
    X_train_tfidf = self.tfidf_transformer.fit_transform(X_train_counts)

    self.classifier = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=1000, random_state=42)

    self.classifier.fit(X_train_tfidf, decisions)

    return test_set

  def test(self, test_set, test_titles=False):
    """
    Method used to test the SDGClassifier.
    """
    articles = []
    decisions = []
    count = 0
    for article in test_set:
      try:
        orgs = Reuters_PMI.find_incident_orgs(article['title'])
        if orgs == []:
          continue
        prices = QuandlWrapper.query_org_prices(orgs[0], QuandlWrapper.convert_dates([article['time_string'], QuandlWrapper.add_week(article['time_string'])]))
      except Exception as e:
        print(traceback.format_exc())
        continue
      articles.append(article['text' if not test_titles else 'title'])
      if prices['close'] > prices['open']:
        decisions.append('positive')
      else:
        decisions.append('negative')
      count += 1
      if count == 3000:
        break  

    counts = self.count_vec.transform(articles)
    test_tfidf = self.tfidf_transformer.transform(counts)
    classification = self.classifier.predict(test_tfidf)
    correct_count = 0
    overall_count = 0
    index = 0
    for decision in classification:
      if decision == decisions[index]:
        correct_count += 1
      overall_count += 1
      index += 1
    print(overall_count)
    print(correct_count / overall_count)




svm = SVM_Trainer(2009)

test_set = svm.train(train_titles=True)

svm.test(test_set, test_titles=True)
  

#nltk_train_semeval()
    
    
#nltk_train('/home/jmkovachi/sent-classifier/movie_reviews/txt_sentoken/')
        
        