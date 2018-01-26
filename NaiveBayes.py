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
from sklearn.metrics import precision_recall_fscore_support as score


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
        self.doc_count = self.pos_count + self.neg_count # + self.neu_count
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
       # self.features['neutralFeatures'] = {}

        # Gathering a priori probabilities by class
        self.priorLogPos = math.log(self.pos_count/self.doc_count)
        self.priorLogNeg = math.log(self.neg_count/self.doc_count)
        #self.priorLogNeutral = math.log(self.neu_count/self.doc_count)
 
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
    def test(self, document):
        wordnet_lemmatizer = WordNetLemmatizer()
        document = [wordnet_lemmatizer.lemmatize(x) for x in document.split(" ")]
        pos_val = self.priorLogPos
        neg_val = self.priorLogNeg
        # neutral_val = self.priorLogNeutral
        
        # Smoothed probabilities are calculated below, these are used when a 
        # word in the test document is not found in the given class but is found
        # in another class's feature dict
        smooth_pos = math.log(1/(self.pos_count + self.doc_count))
        smooth_neg = math.log(1/(self.neg_count + self.doc_count))
        # smooth_neutral = math.log(1/(self.neu_count + self.doc_count))
        
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
            return ('positive', pos_val)
        
        
    """
    Downloads an article from NYT to use for testing.
    """
    @staticmethod
    def eval(domain, obj, text=''):
        import newspaper
        news = newspaper.build(domain, memoize_articles=False, language='en')
        if len(text) == 0:
            paper = news.articles[0]
            paper.download()
            paper.parse()
            article = paper.text
        else:
            article = text

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
       # outcome = bayes_object.test(sentences[i])
        
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
          #  elif sent_array[count] == 'NEU':
          #      annotated_array.append(2)
            else:
                print('Malformed input for sentence: ' + sentences[count])
                count += 1
                continue
            
            if outcome[0] == 'positive':
                predicted_array.append(0)
            elif outcome[0] == 'negative':
                predicted_array.append(1)
           # elif outcome[0] == 'neutral':
           #     predicted_array.append(2)
            
            count += 1
            
        precision, recall, fscore, support = score(annotated_array, predicted_array)
        
        print(annotated_array)
        print(predicted_array)
        
        print('precision: {}'.format(precision))
        print('recall: {}'.format(recall))
        print('fscore: {}'.format(fscore))
        print('support: {}'.format(support))
            
            
        
        """if sent_array[i] == 'X':
            print('Sentence was malformed')
        elif sent_array[i] == 'P':
            if outcome[0] == 'negative':
                counts[0][1] += 1
            elif outcome[0] == 'neutral':
                counts[0][2] += 1
            elif outcome[0] == 'positive':
                counts[0][0] += 1
        elif sent_array[i] == 'N':
            if outcome[0] == 'negative':
                counts[1][1] += 1
            elif outcome[0] == 'neutral':
                counts[1][2] += 1
            elif outcome[0] == 'positive':
                counts[1][0] += 1
        elif sent_array[i] == 'NEU':
            if outcome[0] == 'negative':
                counts[2][1] += 1
            elif outcome[0] == 'neutral':
                counts[2][2] += 1
            elif outcome[0] == 'positive':
                counts[2][0] += 1
        else:
            print('incorrect input')"""
            
    
        
                  
mpqa = MPQA()

#MN_NaiveBayes.eval()




pol_list, sent_list = mpqa.build_BOW(file_path='/home/jmkovachi/sent-classifier/database.mpqa.3.0/gate_anns')

pos, neg, neutral = mpqa.build_counts(pol_list, sent_list)

print("Positive words: " + str(pos))
print("Negative words: " + str(neg))

NB = MN_NaiveBayes(pos, neg, neutral)

NB.train()

#print(NB.test('He condemned him'))
 
NB.eval("https://finance.yahoo.com/news/",NB, """High fuel prices hit BA's profits

British Airways has blamed high fuel prices for a 40% drop in profits.

Reporting its results for the three months to 31 December 2004, the airline made a pre-tax profit of £75m ($141m) compared with £125m a year earlier. Rod Eddington, BA's chief executive, said the results were "respectable" in a third quarter when fuel costs rose by £106m or 47.3%. BA's profits were still better than market expectation of £59m, and it expects a rise in full-year revenues.

To help offset the increased price of aviation fuel, BA last year introduced a fuel surcharge for passengers.

In October, it increased this from £6 to £10 one-way for all long-haul flights, while the short-haul surcharge was raised from £2.50 to £4 a leg. Yet aviation analyst Mike Powell of Dresdner Kleinwort Wasserstein says BA's estimated annual surcharge revenues - £160m - will still be way short of its additional fuel costs - a predicted extra £250m. Turnover for the quarter was up 4.3% to £1.97bn, further benefiting from a rise in cargo revenue. Looking ahead to its full year results to March 2005, BA warned that yields - average revenues per passenger - were expected to decline as it continues to lower prices in the face of competition from low-cost carriers. However, it said sales would be better than previously forecast. "For the year to March 2005, the total revenue outlook is slightly better than previous guidance with a 3% to 3.5% improvement anticipated," BA chairman Martin Broughton said. BA had previously forecast a 2% to 3% rise in full-year revenue.

It also reported on Friday that passenger numbers rose 8.1% in January. Aviation analyst Nick Van den Brul of BNP Paribas described BA's latest quarterly results as "pretty modest". "It is quite good on the revenue side and it shows the impact of fuel surcharges and a positive cargo development, however, operating margins down and cost impact of fuel are very strong," he said. Since the 11 September 2001 attacks in the United States, BA has cut 13,000 jobs as part of a major cost-cutting drive. "Our focus remains on reducing controllable costs and debt whilst continuing to invest in our products," Mr Eddington said. "For example, we have taken delivery of six Airbus A321 aircraft and next month we will start further improvements to our Club World flat beds." BA's shares closed up four pence at 274.5 pence.""")
        