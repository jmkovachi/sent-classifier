#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 19:27:44 2018

@author: jmkovachi
"""

import math
from read_MPQA import extract_MPQA as MPQA
from nltk.stem import WordNetLemmatizer


class MN_NaiveBayes:

    """
    Constructor for MN_NaiveBayes.
    Initializes overall counts of positive, negative, and neutral classes.
    Initializes overall document count for use in a priori class probability
    calculation.
    Initializes pos, neg, and neutral feature count dictionaries.
    """
    def __init__(self, pos, neg, neutral):
        self.pos_count = MPQA.count(pos)
        self.neg_count = MPQA.count(neg)
        self.neu_count = MPQA.count(neutral)
        self.doc_count = self.pos_count + self.neg_count + self.neu_count
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

        # Gathering a priori probabilities by class
        self.priorLogPos = math.log(self.pos_count/self.doc_count)
        self.priorLogNeg = math.log(self.neg_count/self.doc_count)
        self.priorLogNeutral = math.log(self.neu_count/self.doc_count)
 
        """
        Each for loop below is calculating probabilities of each feature
        for each class.
        Backslashes in calculations are added for readiblity and serve as 
        line breaks.
        """
        for word, count in self.pos.items():
            self.features['posFeatures'][word] = math.log((count + 1) \
                                            /(self.pos_count + self.doc_count))
        for word, count in self.neg.items():
            self.features['negFeatures'][word] = math.log((count + 1) \
                                            /(self.neg_count + self.doc_count))

        for word, count in self.neutral.items():
            self.features['neutralFeatures'][word] = math.log((count + 1) \
                                            /(self.neu_count + self.doc_count))

    """
    Takes a given test document and make a classification decision based off
    of a max probability.
    @param document: Test document used to make classification decision.
    return: A two-tuple with the classification decision and its corresponding
    log-space probability.
    """
    def test(self, document):
        wordnet_lemmatizer = WordNetLemmatizer()
        document = [wordnet_lemmatizer.lemmatize(x) for x in document.split(" ")]
        pos_val = self.priorLogPos
        neg_val = self.priorLogNeg
        neutral_val = self.priorLogNeutral
        
        # Smoothed probabilities are calculated below, these are used when a 
        # word in the test document is not found in the given class but is found
        # in another class's feature dict
        smooth_pos = math.log(1/(self.pos_count + self.doc_count))
        smooth_neg = math.log(1/(self.neg_count + self.doc_count))
        smooth_neutral = math.log(1/(self.neu_count + self.doc_count))
        
        for feature in self.features:
            if feature == 'posFeatures':
                for word in document:
                    if word in self.features['posFeatures']:
                        pos_val += self.features['posFeatures'][word]
                    elif word in self.features['negFeatures'] or self.features['neutralFeatures']:
                        pos_val += smooth_pos
            elif feature == 'negFeatures':
                for word in document:
                    if word in self.features['negFeatures']:
                        neg_val += self.features['negFeatures'][word]
                    elif word in self.features['posFeatures'] or self.features['neutralFeatures']:
                        neg_val += smooth_neg
            elif feature == 'neutralFeatures':
                for word in document:
                    if word in self.features['neutralFeatures']:
                        neutral_val += self.features['neutralFeatures'][word]
                    elif word in self.features['posFeatures'] or self.features['negFeatures']:
                        neutral_val += smooth_neutral
        
        
        if pos_val > neg_val and pos_val > neutral_val:
            return ('positive', pos_val)
        elif neg_val > pos_val and neg_val > neutral_val:
            return ('negative', neg_val)
        elif neutral_val > pos_val and neutral_val > neg_val:
            return ('neutral', neutral_val)
        else:
            return ('positive', pos_val)
                  
mpqa = MPQA()

pol_list, sent_list = mpqa.build_BOW(file_path='/home/jmkovachi/sent-classifier/database.mpqa.3.0/gate_anns')

pos, neg, neutral = mpqa.build_counts(pol_list, sent_list)

print(neg)   
NB = MN_NaiveBayes(pos, neg, neutral)

NB.train()

print(NB.test('He urged and supported him on his journey'))
        
        