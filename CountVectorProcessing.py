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
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer,TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import KFold
from sklearn.metrics import confusion_matrix, f1_score,accuracy_score




warnings.filterwarnings("ignore")
documentClass={}
bag_of_words={}
words_in_multiple_conv={}

""" Pipeline works exactly like it names. It performs tasks used a pipeline 
,i.e. by providing output of first step to the next. 
Sequence processsing starts with the first method in pipeline"""
pipeline = Pipeline([
	('count_vectorizer',   CountVectorizer(ngram_range=(1, 2))),
	('tfidf_transformer',   TfidfVectorizer()),
('classifier',         MultinomialNB())
])

"""this method uses preprocessing class . 
While using countvectorizer, if you want to use custom tokenizer,
 this method can be used"""
def tokenize(conversation):
	sentences=conversation.split(".")
	tokenized=PreProcess.tokenize_sentences(sentences)
	filtered=PreProcess.RemovePunctAndStopWords(tokenized)
	return filtered

def merge_two_dicts(x, y):
	"""Given two dicts, merge them into a new dict as a shallow copy."""
	z = x.copy()
	z.update(y)
	return z
	


def initProcessing():
	global documentClass,totalConversations
	
	conversations= pd.read_csv('train_input.csv')
	# selecting only the conversations column 
	conversOnly=conversations[["conversation"]]
	totalConversations=conversOnly.shape[0]
	#remove print statements later
	print conversOnly.head()
	y_data=pd.read_csv('train_output.csv')
	print y_data.head()
	# selecting only the category column 
	y_values=y_data[["category"]]

	""" Using 6 fold cross validation"""
	k_fold = KFold(n=totalConversations, n_folds=6)
	scores = []
	confusion = np.array([[0, 0], [0, 0]])
	for train_indices, test_indices in k_fold:
		train_text = conversOnly.iloc[train_indices]['conversation'].values
		train_y = y_values.iloc[train_indices]['category'].values.astype(str)

		test_text = conversOnly.iloc[test_indices]['conversation'].values
		test_y = y_values.iloc[test_indices]['category'].values.astype(str)
		ScikitLearnMethod(train_text, train_y,test_text,test_y)


		
""" Uses pipeline to count the number of words, 
calculates TF-IDF values and feeds those values to Multinomial naive bayes.
 If you wana change the classifier, change it in the pipeline. Pretty simple."""
def ScikitLearnMethod(train_text, train_y,test_text,test_y):
	pipeline.fit(train_text, train_y)
	predictions = pipeline.predict(test_text)
	print predictions
	#confusion += confusion_matrix(test_y, predictions)
	score = f1_score(test_y, predictions)
	accuracy = accuracy_score(test_y, predictions)
	scores.append(score)
	print score
	print accuracy
	


if __name__ == "__main__":
	initProcessing()