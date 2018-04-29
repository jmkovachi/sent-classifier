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
import QueryES
import TestResults
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
client = MongoClient("mongodb://127.0.0.1:27018")
db = client['primer']


def word_feats(words):
    return dict([(word, True) for word in words])

class Trainer:

  def __init__(self, year=2007):
    self.classifier = None
    self.year = year

class NB_Trainer(Trainer):

    def __init__(self, year=2007):
        self.classifier = None
        Trainer.__init__(self, year)


    def nltk_train_semeval(self):
        """
        This function trains the semeval data on the NLTK naivebayes classifier.
        """
        d = Data()
        data = d.data
    
        words = [ (word_feats(nltk.word_tokenize(obj['title'])), 'positive') for obj in data if obj['sentiment'] > 0.25 ]
        words.extend([ (word_feats(nltk.word_tokenize(obj['title'])), 'negative') for obj in data if obj['sentiment'] < -0.25 ])

        random.shuffle(words)
        train_set = words
        self.classifier = nltk.NaiveBayesClassifier.train(train_set)


    def classify(self, text):
        decision = self.classifier.classify(word_feats(nltk.word_tokenize(text)))
        return decision

    def test(self, test_titles=False):
      """
      Method used to test the NBClassifier.
      """
      db_articles = [article for article in db.articles.find()]
      articles = []
      decisions = []
      count = 0
      for article in db_articles:
        article = dict(article)
        article['text'] = " ".join([lemmatizer.lemmatize(word) for sentence in nltk.sent_tokenize(article['text']) for word in nltk.word_tokenize(sentence)])
        article['title'] = " ".join([lemmatizer.lemmatize(word) for sentence in nltk.sent_tokenize(article['title']) for word in nltk.word_tokenize(sentence)])
        try:
          orgs = Reuters_PMI.find_incident_orgs(article['title'])
          if orgs == []:
            continue
          prices = QuandlWrapper.query_org_prices(orgs[0], QuandlWrapper.convert_dates([article['time_string'], QuandlWrapper.add_week(article['time_string'])]))
        except Exception as e:
          print(traceback.format_exc())
          continue
        if prices['close']/prices['open'] > 1.015:
          decisions.append('positive')
        elif prices['close']/prices['open'] <= .985:
          decisions.append('negative')
        else:
          continue
        articles.append(article['text' if not test_titles else 'title']) 

      index = 0
      true_pos = 0
      false_pos = 0
      true_neg = 0
      false_neg = 0
      for article in articles:
        decision = self.classify(article)
        if decision == decisions[index]:
          if decision == decisions[index] and decision == 'positive':
            true_pos += 1
          elif decision != decisions[index] and decision == 'positive':
            false_pos += 1
          elif decision == decisions[index] and decision == 'negative':
            true_neg += 1
          elif decision != decisions[index] and decision == 'negative':
            false_neg += 1
        index += 1


      return TestResults.TestResults.compute_scores(true_pos, true_neg, false_pos, false_neg)


class SVM_Trainer(Trainer):

  def __init__(self, year=2007, use_mongo_orgs=False):
    self.classifier = None
    self.use_mongo_orgs = use_mongo_orgs
    if self.use_mongo_orgs:
      self.Query_Mongo = QueryES.QueryES(db)
    Trainer.__init__(self, year)

  def train(self, train_titles=False):
    """
    Method to train the SDG classifier.
    @param train_titles: boolean to indicate whether titles or bodies of text should be trained.
    return: test set of articles that we will use for training.
    """
    count_articles = 0

    time = datetime.datetime(2006, 10, 19) #first available date from articles
    articles = []
    decisions = []
    test_set = []

    while time < datetime.datetime(2014, 1, 1):
      time = time + timedelta(days=1)
  
      end_time = time + timedelta(days=1)
  
      count_articles = 0
      train_set = []
      
      for article in db.articles.find({'date': {'$gte': time, '$lt': end_time}}):
        article = dict(article)
        article['text'] = " ".join([lemmatizer.lemmatize(word) for sentence in nltk.sent_tokenize(article['text']) for word in nltk.word_tokenize(sentence)])
        article['title'] = " ".join([lemmatizer.lemmatize(word) for sentence in nltk.sent_tokenize(article['title']) for word in nltk.word_tokenize(sentence)])
        if count_articles % 4 == 0:
          test_set.append(article)
        else:
          train_set.append(article)
        count_articles += 1
      for article in train_set:
        try:
          if article['title'] == '':
            continue
          if self.use_mongo_orgs:
            orgs = self.Query_Mongo.search_db_for_orgs(article['title'])
          else:
            orgs = Reuters_PMI.find_incident_orgs(article['title'])
          if orgs == []:
            continue

          prices = QuandlWrapper.query_org_prices(orgs[0] if not self.use_mongo_orgs else orgs, QuandlWrapper.convert_dates([article['time_string'], QuandlWrapper.add_week(article['time_string'])]))
        except Exception as e:
          print(traceback.format_exc())
          continue
        if prices['close']/prices['open'] > 1.015:
          decisions.append('positive')
        elif prices['close']/prices['open'] < .985:
          decisions.append('negative')
        else:
          continue
        articles.append(article['text' if not train_titles else 'title'])

    decisions = np.array(decisions)
    self.count_vec = CountVectorizer()

    X_train_counts = self.count_vec.fit_transform(articles)

    print(X_train_counts)
    self.tfidf_transformer = TfidfTransformer()
    X_train_tfidf = self.tfidf_transformer.fit_transform(X_train_counts)

    self.classifier = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, n_iter=1000, random_state=42, shuffle=True)

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
        if self.use_mongo_orgs:
          orgs = self.Query_Mongo.search_db_for_orgs(article['title'])
        else:
          orgs = Reuters_PMI.find_incident_orgs(article['title'])
        if orgs == []:
          continue
        prices = QuandlWrapper.query_org_prices(orgs[0], QuandlWrapper.convert_dates([article['time_string'], QuandlWrapper.add_week(article['time_string'])]))
      except Exception as e:
        print(traceback.format_exc())
        continue
      if prices['close']/prices['open'] > 1 :
        decisions.append('positive')
      elif prices['close']/prices['open'] <= 1:
        decisions.append('negative')
      else:
        continue
      articles.append(article['text' if not test_titles else 'title'])
      count += 1


    counts = self.count_vec.transform(articles)
    test_tfidf = self.tfidf_transformer.transform(counts)
    classification = self.classifier.predict(test_tfidf)

    return TestResults.TestResults.test_input(classification, decisions)

        
        