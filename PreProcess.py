""" 
Created 5 Feb, 2017
Author : Inderjot Kaur Ratol
"""

import pandas  as pd
import warnings
import os
from nltk.corpus import stopwords
import re
from nltk.wsd import lesk
from nltk.corpus import wordnet as wn
from itertools import chain
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import pos_tag
import sys
import operator
from nltk.stem.wordnet import WordNetLemmatizer

stoplist = stopwords.words('english')
stoplist.extend(['com', 'imgur', '', 'co', 'gt', 'st', 'th', 'u', 'r'])
stoplist = set(stoplist)


def RemovePunctAndStopWords(tokens):
	nonPunct = re.compile('[A-Za-z]+')  # should only contain letters
	filtered = [w for w in tokens if nonPunct.match(w)]
	filtered=[w for w in filtered if w.isalpha()]
	#Remove the stopwords from filtered text
	filtered_words = [word for word in filtered if word.lower() not in stoplist]
	#add words to this list if you see any unimportant words occurring many times
	frequent_words=['the','and','of','this','\n','on','in','maybe','may']
	"""converting all words into lowercase for 
	easy comparison and checking if the word occurs in frequent words list"""
	filtered_words = [word for word in filtered_words if word.lower() not in frequent_words]
	"""finds lemma of each word and replace the word with lemma"""
	filtered_words=lemmatizeAndPOSTagWords(filtered_words)
	return filtered_words
	
def tokenize_sentences(sentences):
	'''
	Tokenize into words in sentences.
	Returns list of strs
	'''
	retval = []
	for sent in sentences:
		tokens = word_tokenize(sent)
		retval.extend(tokens)
	return retval

def lemmatizeAndPOSTagWords(words):
	lemmas=[]
	for word, tag in pos_tag(words):
		wntag = tag[0].lower()
		wntag = wntag if wntag in ['a', 'r', 'n', 'v'] else None
		if not wntag:
			lemma = word
		else:
			lemma = WordNetLemmatizer().lemmatize(word, wntag)
		lemmas.append(lemma)
	return lemmas
	