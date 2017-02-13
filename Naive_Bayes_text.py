""" 
Created 5 Feb, 2017
Author : Inderjot Kaur Ratol
"""

import csv
import numpy as np
from numpy import genfromtxt
import pandas  as pd
import warnings
import matplotlib.pyplot as plt
from sklearn import cross_validation
import os
import re
import sys
import operator
import PreProcess


categories=['hockey','movies','nba','news','nfl','politics','soccer','worldnews']
warnings.filterwarnings("ignore")
documentClass={}
bag_of_words={}


	
def processConversation(conversation,category):
	global bag_of_words,documentClass
	bag_of_words={}
	sentences=conversation.split(".")
	tokenized=PreProcess.tokenize_sentences(sentences)
	filtered=PreProcess.RemovePunctAndStopWords(tokenized)
	
	for word in filtered:
		if word in bag_of_words:
			bag_of_words[word]=int(bag_of_words[word])+1
		else:
			bag_of_words[word]=1
	total=len(filtered)
	bag_of_words=calculateFrequencies(total)
	if category in documentClass:
			new_dict=merge_two_dicts(documentClass[category],bag_of_words)
			documentClass[category]=new_dict
	else:
		documentClass[category]=bag_of_words
	
def merge_two_dicts(x, y):
	"""Given two dicts, merge them into a new dict as a shallow copy."""
	z = x.copy()
	z.update(y)
	return z
	
def calculateFrequencies(total):
	global bag_of_words
	for word,count in bag_of_words.items():
		freq=count/float(total)
		bag_of_words[word]=round(freq,2)
	return bag_of_words

def initProcessing():
	global documentClass
	conversations= pd.read_csv('train_input.csv')
	# selecting only the conversations column 
	conversOnly=conversations[["conversation"]]
	y_data=pd.read_csv('train_output.csv')
	# selecting only the category column 
	y_values=y_data[["category"]]

	conversOnly = conversOnly[:500]
	y_values = y_values[:500]

	for con,y in zip(conversOnly.values,y_values.values):
		processConversation(con[0],y[0])
		#put break to check if the first conversation was processed as required. remove otherwise
		#break

		print y
	
	getTopWordsInEachCategory()
	#remove later
	return documentClass
	
def getTopWordsInEachCategory():
	global documentClass
	for classCategory, words in documentClass.items():
		sorted_words = sorted(words.items(), key=operator.itemgetter(1),reverse=True)
		documentClass[classCategory]=sorted_words

if __name__ == "__main__":
	initProcessing()