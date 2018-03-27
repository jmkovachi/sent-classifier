#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 19:27:44 2018

@author: jmkovachi
"""

import math
from read_MPQA import extract_MPQA as MPQA
from nltk.stem import WordNetLemmatizer
from nltk import tokenize
import nltk
from sklearn.metrics import precision_recall_fscore_support as score
from read_movie_reviews import read_Movies as movie_reader
from collections import Counter

class MN_NaiveBayes:

    """
    Constructor for MN_NaiveBayes.
    Initializes overall counts of positive, negative, and neutral classes.
    Initializes overall document count for use in a priori class probability
    calculation.
    Initializes pos, neg, and neutral feature count dictionaries.
    """
    def __init__(self, pos, neg):
        self.pos_count = MPQA.count(pos)
        self.neg_count = MPQA.count(neg)
        self.doc_count = self.pos_count + self.neg_count # + self.neu_count
        self.tie_count = 0
        self.pos = pos
        self.neg = neg

    """
    An implementation of Jurafsky's MN Bayes Network
    algorithm.
    """
    def train(self):
        self.features = {}
        self.features['posFeatures'] = {}
        self.features['negFeatures'] = {}
        
        # Gathering a priori probabilities by class
        self.priorLogPos = math.log(self.pos_count/self.doc_count)
        self.priorLogNeg = math.log(self.neg_count/self.doc_count)

        """
        Each for loop below is calculating probabilities of each feature
        for each class.
        Backslashes in calculations are added for readiblity and serve as 
        line breaks.
        """
        for word, count in self.pos.items():
            self.features['posFeatures'][word] = math.log((int(count) + 1) \
                                            /(self.pos_count + self.doc_count))
        for word, count in self.neg.items():
            self.features['negFeatures'][word] = math.log((int(count) + 1) \
                                            /(self.neg_count + self.doc_count))

        """ for word, count in self.neutral.items():
            self.features['neutralFeatures'][word] = math.log((int(count) + 1) \
                                            /(self.neu_count + self.doc_count))"""

    """
    Takes a given test document and make a classification decision based off
    of a max probability.
    @param document: Test document used to make classification decision.
    return: A two-tuple with the classification decision and its corresponding
    log-space probability.
    """
    def test(self, document, POS=False):
        wordnet_lemmatizer = WordNetLemmatizer()
        
        document = [wordnet_lemmatizer.lemmatize(x) for x in document.split(" ")]
        #print(document)
        if POS:
            document = nltk.pos_tag(document)
            #print(document)
            document = [str(word[0] + '-' + word[1]) for word in document]
            #document = [str(nltk.pos_tag(word)[0][1] + '-' + nltk.pos_tag(word)[1]) for word in document]
        
        pos_val = self.priorLogPos
        neg_val = self.priorLogNeg
        
        # Smoothed probabilities are calculated below, these are used when a 
        # word in the test document is not found in the given class but is found
        # in another class's feature dict
        smooth_pos = math.log(1/(self.pos_count + self.doc_count))
        smooth_neg = math.log(1/(self.neg_count + self.doc_count))

        for feature in self.features:
            if feature == 'posFeatures':
                for word in document:
                    if word in self.features['posFeatures']:
                        pos_val += self.features['posFeatures'][word]
                    elif word in self.features['negFeatures']: # or self.features['neutralFeatures']:
                        pos_val += smooth_pos
            elif feature == 'negFeatures':
                for word in document:
                    if word in self.features['negFeatures']:
                        neg_val += self.features['negFeatures'][word]
                    elif word in self.features['posFeatures']: # or self.features['neutralFeatures']:
                        neg_val += smooth_neg
            """elif feature == 'neutralFeatures':
                for word in document:
                    if word in self.features['neutralFeatures']:
                        neutral_val += self.features['neutralFeatures'][word]
                    elif word in self.features['posFeatures']: # or self.features['negFeatures']:
                        neutral_val += smooth_neutral"""
        
        
        if pos_val > neg_val: # and pos_val > neutral_val:
            return ('positive', pos_val)
        elif neg_val > pos_val: # and neg_val > neutral_val:
            return ('negative', neg_val)
        #elif neutral_val > pos_val and neutral_val > neg_val:
            #return ('neutral', neutral_val)
        else:
            self.tie_count += 1
            return ('positive', pos_val)
        
        
    """
    Downloads an article from NYT to use for testing.
    """
    @staticmethod
    def eval(domain, obj, text='', newspaper=False, pos_test_docs = [], neg_test_docs = [], POS=False):
        if not newspaper:                     
                annotated_array = []
                predicted_array = []
                for doc in pos_test_docs:
                    for sentence in tokenize.sent_tokenize(doc):
                        try:
                            outcome = obj.test(sentence, POS=POS)
                        except Exception as e:
                            print('bad input: ' + sentence)
                            continue
                        annotated_array.append(0)
                        predicted_array.append(0 if outcome[0] == 'positive' else 1)
                
                print(len(neg_test_docs))
                for doc in neg_test_docs:
                    for sentence in tokenize.sent_tokenize(doc):
                        try:
                            outcome = obj.test(sentence, POS=POS)
                        except Exception as e:
                            print('bad input: ' + sentence)
                            continue
                        annotated_array.append(1)
                        predicted_array.append(1 if outcome[0] == 'negative' else 0)
                        
                precision, recall, fscore, support = score(annotated_array, predicted_array)
        
                print(annotated_array)
                print(predicted_array)
                
                print('precision: {}'.format(precision))
                print('recall: {}'.format(recall))
                print('fscore: {}'.format(fscore))
                print('support: {}'.format(support))               
                
        else:
            
            import newspaper
            news = newspaper.build(domain, memoize_articles=False, language='en')
            if len(text) == 0:
                paper = news.articles[0]
                paper.download()
                paper.parse()
                article = paper.texts
            
            article = tokenize.sent_tokenize(article)
            
            print(article)
            sent_array = []
            
            for sentence in article:
                print(sentence)
                answer = input('What sentiment does this sentence possess? (P|N|NEU|X) ')
                sent_array.append(answer)
             
            MN_NaiveBayes.assign(article, sent_array, obj)    
        


    @staticmethod
    def assign(sentences, sent_array, bayes_object):
        count = 0
        annotated_array = []
        predicted_array = []
        for sentence in sentences:
            
            outcome = bayes_object.test(sentence)
            
            if sent_array[count] == 'X':
                count += 1
                continue
            elif sent_array[count] == 'P':
                annotated_array.append(0)
            elif sent_array[count] == 'N':
                annotated_array.append(1)
            else:
                print('Malformed input for sentence: ' + sentences[count])
                count += 1
                continue
            
            if outcome[0] == 'positive':
                predicted_array.append(0)
            elif outcome[0] == 'negative':
                predicted_array.append(1)
 
            count += 1
            
        precision, recall, fscore, support = score(annotated_array, predicted_array)
        
        
        print('precision: {}'.format(precision))
        print('recall: {}'.format(recall))
        print('fscore: {}'.format(fscore))
        print('support: {}'.format(support))
        
    @staticmethod    
    def count_POS(words):
        cnt_pos = Counter()
        cnt_neg = Counter()
        #print(words)
        pos = [str(word[0][0] + '-' + word[0][1]) for word in words if word[1] == 'positive']
        
        neg = [str(word[0][0] + '-' + word[0][1]) for word in words if word[1] == 'negative']
        
        for word in pos:
            cnt_pos[word] += 1
            
        for word in neg:
            cnt_neg[word] += 1
            
        return cnt_pos, cnt_neg
        
    
        
                  
mpqa = MPQA()


#pol_list, sent_list, pos_test_docs, neg_test_docs = movie_reader.read_for_bayes('/home/jmkovachi/sent-classifier/movie_reviews/txt_sentoken/')



#pos, neg = mpqa.build_counts(pol_list, sent_list)

words, pos_test_docs, neg_test_docs = movie_reader.tag_dir('/home/jmkovachi/sent-classifier/movie_reviews/txt_sentoken/')

#print(words)

pos, neg = MN_NaiveBayes.count_POS(words)

NB = MN_NaiveBayes(pos, neg)

NB.train()


# The first couple parameters below are irrelevant
NB.eval("irrelevant", NB, text='', newspaper=False, pos_test_docs=pos_test_docs, neg_test_docs=neg_test_docs, POS=True)

print(NB.tie_count)