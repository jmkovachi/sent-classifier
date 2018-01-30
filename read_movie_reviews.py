#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 28 15:00:01 2018

@author: jmkovachi
"""

import os
import nltk
from nltk.corpus import stopwords
from nltk.probability import FreqDist

class read_Movies:

    @staticmethod
    def read_dir(base_dir):
        feature_list = []
        
        doc_list = []
        
        base_dir = '/home/jmkovachi/sent-classifier/movie_reviews/txt_sentoken/'
        
        stops = set(stopwords.words('english'))
        print(stops)
        for pos in os.listdir(base_dir + 'pos'):
            file = open(base_dir + 'pos/' + pos)
            text = file.read()
            for sentence in nltk.sent_tokenize(text):
                split_sentence = [word for word in nltk.word_tokenize(sentence) if word not in stops]
                for word in split_sentence:
                    if "\n" in word:
                        word = word.replace('\n','')
                    elif word == '' or word == '!' or word == ';' or word == '?' or word == ';':
                        continue
                    feature_list.append(word)
                    
                
                doc_list.append([set(split_sentence), 'positive', {'positive' : 1, 'negative' : 1}])
                
        for neg in os.listdir(base_dir + 'neg'):
            file = open(base_dir + 'neg/' + neg)
            text = file.read()
            for sentence in nltk.sent_tokenize(text):
                split_sentence = [word for word in nltk.word_tokenize(sentence) if word not in stops]
                for word in split_sentence:
                    if "\n" in word:
                        word = word.replace('\n','')
                    elif word == '' or word == '!' or word == ';' or word == '?' or word == ';':
                        continue 
                    feature_list.append(word)
                doc_list.append([set(sentence.split(' ')), 'negative', {'positive' : 1, 'negative' : 1}])
        
        feature_list = FreqDist(feature_list)
        
        new_list = []
        for word, frequency in feature_list.most_common(30):
            new_list.append(word)
            
        return list(set(new_list)), doc_list
    
    @staticmethod
    def read_for_bayes(base_dir):
        pos_list = []
        neg_list = []
        
        pos_index_count = 0
        test_docs_pos = []
        for pos in os.listdir(base_dir + 'pos'):
            file = open(base_dir + 'pos/' + pos)
            text = file.read()
            if pos_index_count / len(os.listdir(base_dir + 'pos')) > .75:
                test_docs_pos.append(text)
                continue
            pos_index_count += 1
            for sentence in text.split('.'):
                stops = set(stopwords.words('english'))
                split_sentence = [word for word in sentence.split(' ') if word not in stops]
                for word in split_sentence:
                    pos_list.append((word, 'positive'))
            
    
        neg_index_count = 0
        test_docs_neg = []
        for neg in os.listdir(base_dir + 'neg'):
            file = open(base_dir + 'neg/' + neg)
            text = file.read()
            if neg_index_count / len(os.listdir(base_dir + 'neg')) > .75:
                test_docs_neg.append(text)
                continue
            neg_index_count += 1
            for sentence in text.split('.'):
                for word in sentence.split(' '):
                    neg_list.append((word, 'negative'))
            
        
        return pos_list, neg_list, test_docs_pos, test_docs_neg
    

        
        