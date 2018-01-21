#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 19:27:44 2018

@author: jmkovachi
"""

from read_MPQA import extract_MPQA as MPQA
import math

class MN_NaiveBayes:
    
    
    def __init__(self, pos, neg, neutral):
        self.pos_count = MPQA.count(pos)
        self.neg_count = MPQA.count(neg)
        self.neu_count = MPQA.count(neutral)
        self.doc_count = self.pos_count + self.neg_count + self.neu_count
        self.key_dict = {x for x in pos.update(neg).update(neutral)}
        self.pos = pos
        self.neg = neg
        self.neutral = neutral
    
    """
    An implementation of Jurafsky's MN Bayes Network 
    algorithm.
    """
    def train(self):
        self.features = {}
        self.features['posFeatures'] = {}
        self.features['negFeatures'] = {}
        self.features['neutralFeatures'] = {}
        
        self.priorLogPos = math.log(self.pos_count/self.doc_count)
        self.priorLogNeg = math.log(self.neg_count/self.doc_count)
        self.priorLogNeutral = math.log(self.neu_count/self.doc_count)
        
        for word, count in self.pos.items():
            self.features['posFeatures'][word] = math.log((count + 1)
                                            /(self.pos_count + self.doc_count))
        for word, count in self.neg.items():
            self.features['negFeatures'][word] = math.log((count + 1)
                                            /(self.neg_count + self.doc_count))
        
        for word, count in self.neutral.items():
            self.features['neutralFeatures'][word] = math.log((count + 1)
                                            /(self.neu_count + self.doc_count))
        
    def test(self,document):
        pos_val = self.priorLogPos
        smooth_pos = math.log(1/(self.pos_count + self.doc_count))
        neg_val = self.priorLogNeg
        smooth_neg = math.log(1/(self.neg_count + self.doc_count))
        neutral_val = self.priorLogNeutral
        smooth_neutral = math.log(1/(self.neu_count + self.doc_count))
        for feature in self.features:
            if feature == 'posFeatures':
                for word in document:
                    if word in self.features['posFeatures']:
                        pos_val += self.features['posFeatures'][word]
                    elif word in self.key_dict:
                        pos_val += smooth_pos
            elif feature == 'negFeatures':
                for word in document:
                    if word in self.features['negFeatures']:
                        neg_val += self.features['negFeatures'][word]
                    elif word in self.key_dict:
                        neg_val += smooth_neg
            elif feature == 'neutralFeatures':
                for word in document:
                    if word in self.features['neutralFeatures']:
                        neutral_val += self.features['neutralFeatures'][word]
                    elif word in self.key_dict:
                        neutral_val += smooth_neutral
        
        if pos_val > neg_val and pos_val > neutral_val:
            return ('positive', pos_val)
        elif neg_val > pos_val and neg_val > neutral_val:
            return ('negative', neg_val)
        elif neutral_val > pos_val and neutral_val > neg_val:
            return ('neutral', neutral_val)
        else:
            return ('positive', pos_val)
                    


        
        
        