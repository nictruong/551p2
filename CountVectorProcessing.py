""" 
Created 5 Feb, 2017
Author : Inderjot Kaur Ratol
"""
import numpy as np
import pandas  as pd
import warnings
import matplotlib.pyplot as plt
import csv
from textblob import TextBlob
import os
import csv
import re
import sys
import operator
import PreProcess
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier 
from sklearn.learning_curve import learning_curve
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import KFold
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix




warnings.filterwarnings("ignore")
documentClass={}
bag_of_words={}
words_in_multiple_conv={}

"""this method uses preprocessing class . 
While using countvectorizer, if you want to use custom tokenizer,
 this method can be used"""
def split_into_lemmas(message):
	message = unicode(message, 'utf8').lower()
	words = TextBlob(message).words
	# for each word, take its "base form" = lemma 
	return [word.lemma for word in words]

""" Pipeline works exactly like it names. It performs tasks used a pipeline 
,i.e. by providing output of first step to the next. 
Sequence processsing starts with the first method in pipeline"""
pipeline = Pipeline([
	('count_vectorizer',CountVectorizer(ngram_range =(1,2),stop_words='english')),
('classifier', LinearSVC())
])




def initProcessing():
	global documentClass,totalConversations
	test_data=pd.read_csv('test_input.csv')
	conversations= pd.read_csv('train_input.csv')
	# selecting only the conversations column 
	conversOnly=conversations["conversation"]
	totalConversations=conversOnly.shape[0]
	#remove print statements later
	print conversOnly.head()
	y_data=pd.read_csv('train_output.csv')
	print y_data.head()
	# selecting only the category column 
	y_values=y_data["category"]
	print test_data['id'].head()
	print test_data['conversation'].head()
	#ScikitLearnMethod(conversOnly.values, y_values.values,test_data['conversation'].values,test_data['id'].values)
	""" Using 6 fold cross validation"""
	k_fold = KFold(n=totalConversations, n_folds=6)
	scores = []
	confusion = np.array([[0, 0], [0, 0]])
	for train_indices, test_indices in k_fold:
		train_text = conversOnly.iloc[train_indices].values
		train_y = y_values.iloc[train_indices].values

		test_text = conversOnly.iloc[test_indices].values
		test_y = y_values.iloc[test_indices].values
		ScikitLearnMethod(train_text, train_y,test_text,test_y)
		break


		
""" Uses pipeline to count the number of words, 
calculates TF-IDF values and feeds those values to Multinomial naive bayes.
 If you wana change the classifier, change it in the pipeline. Pretty simple."""
def ScikitLearnMethod(train_text, train_y,test_text,test_y):
	pipeline.fit(train_text, train_y)
	predictions = pipeline.predict(test_text)
	print predictions
	#writePredictions(predictions,test_y)
	
	#confusion += confusion_matrix(test_y, predictions)
	score = f1_score(test_y, predictions)
	accuracy = accuracy_score(test_y, predictions)
	report=classification_report(test_y, predictions)
	print score
	print accuracy
	print report
	
def writePredictions(predictions,test_y):
	with open('predictionsSVC.csv', 'wb') as out:
		writer = csv.writer(out)
		fieldnames = [('id','category')]
		writer.writerow(fieldnames[0])
		for id,category in zip(test_y,predictions):
			print "writing in file", id, category
			writer.writerow([id, category])

if __name__ == "__main__":
	initProcessing()