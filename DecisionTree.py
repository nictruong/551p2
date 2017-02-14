from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import csv
import re
import numpy as np
import string
import math
import PreProcess

HOCKEY = "hockey"
MOVIES = "movies"
NBA = "nba"
NEWS = "news"
NFL = "nfl"
POLITICS = "politics"
SOCCER = "soccer"
WORLDNEWS = "worldnews"

####################################################################
# Node object for building the decision tree
####################################################################
class Node:
	def __init__(self, leftNode, rightNode, word, columnIndex, topicCounts, depth):
		self.leftNode = leftNode
		self.rightNode = rightNode
		self.word = word # feature on which the set as split on
		self.columnIndex = columnIndex
		self.topicCounts = topicCounts
		self.depth = depth
####################################################################

# Divide set into 2 sets, depending on if their conversation contains a specific word or not
def divideSet(rows, columnIndex):
	set1 = [ row for row in rows if row[ columnIndex ] == 0 ]
	set2 = [ row for row in rows if row[ columnIndex ] == 1 ]

	return set1, set2


# Get number of entries per topic
def getTopicCounts(rows):

	topicCounts = {}

	for row in rows:
		topic = row[ len(row) - 1 ]

		if topic in topicCounts: 
			topicCounts[topic] += 1
		else:
			topicCounts[topic] = 1

	return topicCounts


# Decision Trees: slide 21
def getEntropy(rows):

	h = 0.0

	topicCounts = getTopicCounts(rows)
	nbOfRows = len(rows)

	for topic in topicCounts.keys():

		proportion = float(topicCounts[topic]) / float(nbOfRows)
		try:
			h -= proportion * math.log(proportion, 2)
		except ValueError:
			pass

	return h

# get the best split depending on which column and information gain
def getBestSplit(rows, mostCommonWords):

	bestInformationGain = 0.0
	bestWord = None
	bestSet1 = []
	bestSet2 = []
	bestColumnIndex = -1

	H = getEntropy(rows)

	nbOfColumns = len(mostCommonWords)

	for columnIndex in range(0, nbOfColumns):

		# Split the training set
		# set1 is always 0, set2 is always 1
		(set1, set2) = divideSet(rows, columnIndex)

		# IG: slide 27 step 3
		p = float(len(set1)) / float(len(rows))
		ig = H - (p * getEntropy(set1) + ( 1-p ) * getEntropy(set2))

		# slide 27 step 2
		if ig > bestInformationGain and len(set1) > 0 and len(set2) > 0:
			bestInformationGain = ig
			bestWord = mostCommonWords[columnIndex]
			bestSet1 = set1
			bestSet2 = set2
			bestColumnIndex = columnIndex

	return (bestInformationGain, bestWord, bestColumnIndex, bestSet1, bestSet2)


# Decision trees: slide 28
# Recursive algorithm
def buildTree(rows, mostCommonWords, depth):

	if len(rows) == 0:
		return Node()

	(bestInformationGain, bestWord, bestColumnIndex, bestSet1, bestSet2) = getBestSplit(rows, mostCommonWords)

	if depth == 50:

		topicCount = getTopicCounts(rows)

		dominantTopic = None
		dominantTopicCount = 0
		for topic in topicCount:
			if topicCount[topic] > dominantTopicCount:
				dominantTopic = topic
				dominantTopicCount = topicCount[topic]

		bestTopic = {}
		bestTopic[dominantTopic] = dominantTopicCount

		print bestTopic

		return Node(None, None, None, -1, bestTopic, depth)


	# slide 27 step 1 and 4.
	# If IG is 0, then is a leaf. Else, is a node and recursively repeat
	if bestInformationGain == 0.0:
		return Node(None, None, None, -1, getTopicCounts(rows), depth)
	else:
		leftNode = buildTree(bestSet1, mostCommonWords, depth + 1) 
		rightNode = buildTree(bestSet2, mostCommonWords, depth + 1)
		return Node(leftNode, rightNode, bestWord, bestColumnIndex, None, depth)

def predict(tree, conversation):

	if tree.topicCounts != None:
		return tree.topicCounts.keys()[0]
	else:
		value = conversation[tree.columnIndex]

		branch = None
		if value == 0:
			branch = tree.leftNode
		elif value == 1:
			branch = tree.rightNode

		return predict(branch, conversation)

def kFoldTesting(input, nbOfFolds):

	nbEntries = len(input)
	interval = nbEntries / nbOfFolds	

	for i in range(nbOfFolds):

		start = interval * i
		end = interval * (i + 1)

		training1 = input[0:start]
		training2 = input[end:]
		training = training1 + training2

		validation = input[start:end]	

		mostCommonWords = findMostCommonWords(training)
		parsedTraining = parseInput(training, mostCommonWords)
		parsedValidation = parseInput(validation, mostCommonWords)

		tree = buildTree(parsedTraining, mostCommonWords, 0)

		truePos = 0
		falsePos = 0
		trueNeg = 0
		falseNeg = 0

		for conversation in parsedValidation:

			conversationTopic = conversation[ len(conversation) - 1 ]

			prediction = predict(tree, conversation)

			if prediction == conversationTopic:
				truePos += 1

		print float(truePos) / float(len(parsedValidation)) * 100

def finalPrediction(input, testInput):

	mostCommonWords = findMostCommonWords(input)
	parsedInput = parseInput(input, mostCommonWords)
	parsedTestInput = parseInput(testInput, mostCommonWords)

	print "Building tree..."

	tree = buildTree(parsedInput, mostCommonWords)

	print "Done building tree..."

	predictions = []

	for testConversation, i in zip(parsedTestInput, range(len(parsedTestInput))):

		print "Predicting conversation: " + str(i)

		prediction = predict(tree, testConversation)
		predictionRow = [i, prediction]

		predictions.append(predictionRow)

	with open("output.csv", "wt") as f:
		writer = csv.writer(f)
		writer.writerow(["id", "category"])
		writer.writerows(predictions)


def parseInput(rawInput, words):

	print "Parsing input..."

	parsedInput = []

	for entry in rawInput:
		
		entryVector = [0] * len(words)
		entryWords = cleanData(entry[1])

		for word in entryWords:

			try:
				index = words.index(word)
				entryVector[index] = 1
			except ValueError:
				pass

		# append the actual topic of the conversation to the vector
		entryVector.append(entry[ len(entry) - 1 ])
		parsedInput.append(entryVector)

	print "Done parsing input..."

	return parsedInput

def cleanData(conversation):

	# List of all punctuation
	punctuations = list(string.punctuation)
	punctuations.append("''")

	# stop word set
	stopSet = stopwords.words('english')
	stopSet.extend(['com', 'imgur', '', 'co', 'gt', 'st', 'th', 'u', 'r', 'the','and','of','this','\n','on','in','maybe','may'])

	a = re.sub(r'<(\S)*>\s', '', conversation)
	a = word_tokenize(a)
	a = [i.strip("".join(punctuations)) for i in a if i not in punctuations]
	a = [word for word in a if word not in stopSet]
	# a = PreProcess.lemmatizeAndPOSTagWords(a)

	return a

def findMostCommonWords(data):

	print "Finding most common words..."

	# nb of most frequent words
	frequencyNumber = 150

	bagOfWords = {}
	mostCommonWords = []

	for conversation in data:

		a = cleanData(conversation[1])

		topic = conversation[ len(conversation) - 1 ]

		if topic in bagOfWords: 
			bagOfWords[topic].extend(a)
		else:
			bagOfWords[topic] = a

	for topic in bagOfWords.keys():
		bagOfWords[topic] = FreqDist(word.lower() for word in bagOfWords[topic])
		bagOfWords[topic] = [ x[0] for x in bagOfWords[topic].most_common(frequencyNumber) ]		
		mostCommonWords.extend(bagOfWords[topic])

	print "Done finding most common words..."

	return list(set(mostCommonWords))

def main():

	data = []
	result = []
	testData = []

	with open('train_input.csv','rt') as csvfile:
	    reader = csv.reader(csvfile)
	    for row in reader:
	        data.append(row)

	with open('test_input.csv','rt') as csvfile:
	    reader = csv.reader(csvfile)
	    for row in reader:
	        testData.append(row)

	# with open('train_output.csv','rt') as csvfile:
	#     reader = csv.reader(csvfile)
	#     for row in reader:
	#        result.append(row)	

	data = data[1:]
	testData = testData[1:]
	# result = result[1:5000]

	######################################################
	# KFOLD TESTING
	######################################################
	nbOfFolds = 4

	kFoldTesting(data, nbOfFolds)

	######################################################
	# PREDICTION
	######################################################
	# print "Starting..."

	# finalPrediction(data, testData)

if __name__ == "__main__":
	main()