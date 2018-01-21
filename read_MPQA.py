#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 20 15:30:22 2018

@author: jmkovachi
"""

from xml.etree.ElementTree import ElementTree
import re
import os
from collections import Counter
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

#print(os.listdir('/home/jmkovachi/market-sent-analysis/database.mpqa.3.0/gate_anns'))
class extract_MPQA:
    
    def build_BOW(self, file_path):
        pol_list = []
        sent_list = []
        count = 0
        count_errors = 0
        for directory in os.listdir(file_path):
            count += 1
            try:
                for sub_dir in os.listdir(file_path + '/' + directory):
                    if (sub_dir == '.DS_Store'): continue
                    for xml in os.listdir(file_path + '/' + directory + '/' + sub_dir):
                        tree = ElementTree()
                        path = file_path + '/' + directory + '/' + sub_dir + '/' + xml
                        try:
                            tree.parse(path)
                        except Exception as e:
                            print("Problem: " + xml)
                            continue
                        xml_string = ''
                        file = open(path, 'r')
                        xml_string = file.read() 
                        annotation = list(tree.iter('Annotation'))
                        for ann in annotation:
                            features = list(ann.iter('Feature'))
                            ann_items = ann.attrib
                            for f in features:
                               name = f.find('Name')
                               value = f.find('Value')
                               if name.text == 'polarity':
                                   start = ann_items['StartNode']
                                  #end = ann_items['EndNode']
                                   try:
                                       find_text = re.search('id="' + start + '" />(.+?)<Node', xml_string).group(1)
                                       pol_list.append((find_text.strip().lower(), value.text))
                                   except Exception as e:
                                       #print(e)
                                       count_errors += 1
                                       #print(count_errors)
                               elif name.text == 'attitude-type':
                                   start = ann_items['StartNode']
                                   #end = ann_items['EndNode']
                                   try:
                                       find_text = re.search('id="' + start + '" />(.+?)<Node', xml_string).group(1)
                                       sent_list.append((find_text.strip().lower(), value.text))
                                   except Exception as e:
                                       count_errors += 1
                                   
        
            except Exception as e:
                print(str(e) + " " + str(count))
                continue
        return pol_list, sent_list
        
    
    
    def build_counts(self, pol_list, sent_list):
        wordnet_lemmatizer = WordNetLemmatizer()
        cnt_pos = Counter()
        cnt_neg = Counter()
        cnt_neu = Counter()
        pol_list.extend(sent_list)
        stops = set(stopwords.words('english'))
        for pol in pol_list:
            #print(pol[1])

            pos = [wordnet_lemmatizer.lemmatize(x) for x in pol[0].split(' ') 
                    if (pol[1] == 'positive' or pol[1] == 'uncertain-positive' 
                        or pol[1] == 'both' or pol[1] == 'sentiment-pos') 
                    and x not in stops]
            neg = [wordnet_lemmatizer.lemmatize(x) for x in pol[0].split(' ') 
                    if (pol[1] == 'negative' or pol[1] == 'uncertain-negative' 
                        or pol[1] == 'both' or pol[1] == 'sentiment-neg') 
                    and x not in stops] 
            neu = [wordnet_lemmatizer.lemmatize(x) for x in pol[0].split(' ') 
                    if (pol[1] == 'neutral')
                    and x not in stops] 
            
            for word in pos:
                cnt_pos[word] += 1
            for word in neg:
                cnt_neg[word] += 1
            for word in neu:
                cnt_neu[word] += 1
        return cnt_pos, cnt_neg, cnt_neu
    
    def count(self, pol_dict):
        count = 0
        for entry in pol_dict:
            count += entry
        return count
   
MPQA = extract_MPQA()


pol_list, sent_list = MPQA.build_BOW('/home/jmkovachi/sent-classifier/database.mpqa.3.0/gate_anns')


#print(pol_list)
#print(sent_list)
pos, neg, neu = MPQA.build_counts(pol_list,sent_list)

print(neg)

wordnet_lemmatizer = WordNetLemmatizer()

print(wordnet_lemmatizer.lemmatize('supported'))
"""
    
print(len(final_str))          

print(count_errors)

print(len(final_str.split('\n')))

write_file = open('lexicon.txt','w')
write_file.write(final_str)
write_file.close()

write_file2 = open('lexicon2.txt','w')
write_file2.write(final_str2)
write_file2.close()


print(len(final_str2))          


print(len(final_str2.split('\n')))

"""
