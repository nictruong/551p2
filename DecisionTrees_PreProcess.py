""" 
Created 5 Feb, 2017
Author : Inderjot Kaur Ratol
"""
import numpy as np
import pandas  as pd
import warnings
import matplotlib.pyplot as plt
import os
import re
import sys
import operator
import PreProcess

from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix, f1_score,accuracy_score

import DecisionTree



categories=['hockey','movies','nba','news','nfl','politics','soccer','worldnews']
warnings.filterwarnings("ignore")
documentClass={}
bag_of_words={}
words_in_multiple_conv={}





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
	#total=len(filtered)
	#bag_of_words=calculateFrequencies(total)
	addTermFrequency(bag_of_words)
	
	
def addWordsInClassCategory(category):
	global documentClass,bag_of_words
	if category in documentClass:
		new_dict=merge_two_dicts(documentClass[category],bag_of_words)
		documentClass[category]=new_dict
	else:
		documentClass[category]=bag_of_words
		
def addTermFrequency(words):
	global words_in_multiple_conv
	for word in words:
		if word in words_in_multiple_conv:
			words_in_multiple_conv[word]=int(words_in_multiple_conv[word])+1
		else:
			words_in_multiple_conv[word]=1

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
	global documentClass,totalConversations
	
	conversations= pd.read_csv('train_input.csv')
	# selecting only the conversations column 
	conversOnly=conversations[["conversation", "category"]]
	totalConversations=conversOnly.shape[0]
	y_data=pd.read_csv('train_output.csv')
	# selecting only the category column 
	y_values=y_data[["category"]]

	conversOnly = conversOnly[:5000]
	# y_values = y_values[:1000]
	totalConversations=conversOnly.shape[0]

	k_fold = KFold(n=totalConversations, n_folds=6)
	scores = []

	for train_indices, test_indices in k_fold:

		train = conversOnly.iloc[train_indices].values
		# train_y = y_values.iloc[train_indices]['category'].values.astype(str)

		test = conversOnly.iloc[test_indices].values
		# test_y = y_values.iloc[test_indices]['category'].values.astype(str)

		DecisionTrees(train, test)

	
def DecisionTrees(train, test):
	global documentClass

	frequencyNumber = 100

	count=0
	documentClass={}
	for con, y in train:
		count=count+1
		print "processing conversation number",count,"\n"
		processConversation(con, y)
		addWordsInClassCategory(y)
		#put break to check if the first conversation was processed as required. remove otherwise
		#break
	# applyTFIDF()
	getTopWordsInEachCategory()

	mostCommonWords = []

	# print documentClass['nfl']
	# print '###############################################'
	# print '###############################################'
	# print '###############################################'
	# print '###############################################'
	# print '###############################################'

	# for key in documentClass.keys():

	# 	topWords = []

	# 	for i in range(frequencyNumber):

	# 		topWords.append(documentClass[key][i][0])

	# 	mostCommonWords.extend(topWords)

	# parsedTraining = DecisionTree.parseInput(train, mostCommonWords)
	# parsedValidation = DecisionTree.parseInput(test, mostCommonWords)

	# tree = DecisionTree.buildTree(parsedTraining, mostCommonWords)

	# truePos = 0

	# for conversation in parsedValidation:

	# 	conversationTopic = conversation[ len(conversation) - 1 ]

	# 	prediction = DecisionTree.predict(tree, conversation)

	# 	if prediction == conversationTopic:
	# 		truePos += 1

	# print float(truePos) / float(len(parsedValidation)) * 100



def applyTFIDF():
	global documentClass,words_in_multiple_conv,totalConversations
	for classCategory, words in documentClass.items():
		for word,count in words.items():
			if word in words_in_multiple_conv:
				termFrequency=words_in_multiple_conv[word]
				idf=np.log(totalConversations/float(termFrequency+1))
				tfidf=count*idf
				words[word]=tfidf
				
def getTopWordsInEachCategory():
	global documentClass
	for classCategory, words in documentClass.items():
		sorted_words = sorted(words.items(), key=operator.itemgetter(1),reverse=True)
		documentClass[classCategory]=sorted_words

if __name__ == "__main__":
	initProcessing()