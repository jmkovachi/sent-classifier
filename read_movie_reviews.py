#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 15:00:01 2018

@author: jmkovachi
"""

import os
from nltk.corpus import stopwords
from nltk.probability import FreqDist

class read_Movies:

    @staticmethod
    def read_dir(base_dir):
        feature_list = []
        
        doc_list = []
        
        base_dir = '/home/jmkovachi/sent-classifier/movie_reviews/txt_sentoken/'
        
        for pos in os.listdir(base_dir + 'pos'):
            file = open(base_dir + 'pos/' + pos)
            text = file.read()
            for sentence in text.split('.'):
                stops = set(stopwords.words('english'))
                split_sentence = [word for word in sentence.split(' ') if word not in stops]
                for word in split_sentence:
                    feature_list.append(word)
                    
                
                doc_list.append([set(split_sentence), 'positive', {'positive' : 1, 'negative' : 1}])
                
        for neg in os.listdir(base_dir + 'neg'):
            file = open(base_dir + 'neg/' + neg)
            text = file.read()
            for sentence in text.split('.'):
                for word in sentence.split(' '):
                    feature_list.append(word)
                doc_list.append([set(sentence.split(' ')), 'negative', {'positive' : 1, 'negative' : 1}])
        
        feature_list = FreqDist(feature_list)
        
        new_list = []
        for word, frequency in feature_list.most_common(1000):
            new_list.append(word)
            
        return list(set(new_list)), doc_list