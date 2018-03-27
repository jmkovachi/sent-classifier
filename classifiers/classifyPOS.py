#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  4 23:39:36 2018

@author: jmkovachi
"""

from read_movie_reviews import read_Movies as movie_reader
import nltk
import random

def nltk_train(base_dir):
    
    words = movie_reader.tag_dir(base_dir)
    random.shuffle(words)
    train_set, test_set = words[0:(len(words)*3/4)], words[(len(words)*3/4):]
    
    classifier = nltk.NaiveBayesClassifier.train(train_set)
    
    print(nltk.classify.accuracy(classifier, test_set))
    
    
nltk_train('/home/jmkovachi/sent-classifier/movie_reviews/txt_sentoken/')
        
        