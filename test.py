from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.probability import FreqDist
import csv
import re
import numpy as np
import string

HOCKEY = "hockey"
MOVIES = "movies"
NBA = "nba"
NEWS = "news"
NFL = "nfl"
POLITICS = "politics"
SOCCER = "soccer"
WORLDNEWS = "worldnews"

def divideSet(rows, column):
	set1 = [ row for row in rows if row[column] == 1]
	set2 = [ row for row in rows if row[column] == 0]

	return set1, set2

def uniqueCounts(rows):
	sum0 = 0
	sum1 = 0

	for entry in


def entropy(rows):

	print rows




def decisionTree(input):

	entropy(input[0])


def parseInput(rawInput, words):

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

		parsedInput.append(entryVector)

	return parsedInput

def cleanData(conversation):
	# List of all punctuation
	punctuations = list(string.punctuation)
	punctuations.append("''")

	# stop word set
	stopSet = stopwords.words('english')
	stopSet.extend(['com', ''])

	a = re.sub(r'<(\S)*>\s', '', conversation)
	a = word_tokenize(a)
	a = [i.strip("".join(punctuations)) for i in a if i not in punctuations]
	a = [word for word in a if word not in stopSet]

	return a

def main():

	# hockey, movies, nba, news, nfl, politics, soccer, worldnews

	# nb of most frequent words
	frequencyNumber = 10

	data = []
	result = []

	with open('train_input.csv','rt') as csvfile:
	    reader = csv.reader(csvfile)
	    for row in reader:
	        data.append(row)

	with open('train_output.csv','rt') as csvfile:
	    reader = csv.reader(csvfile)
	    for row in reader:
	        result.append(row)	

	data = data[1:100]
	result = result[1:100]

	hockeyWords = []
	moviesWords = []
	nbaWords = []
	newsWords = []
	nflWords = []
	politicsWords = []
	soccerWords = []
	worldNewsWords = []

	for conversation, result in zip(data, result):

		a = cleanData(conversation[1])

		topic = result[1]

		if (topic == HOCKEY):
			hockeyWords.extend(a)
		elif (topic == MOVIES):
			moviesWords.extend(a)
		elif (topic == NBA):
			nbaWords.extend(a)
		elif (topic == NEWS):
			newsWords.extend(a)
		elif (topic == NFL):
			nflWords.extend(a)
		elif (topic == POLITICS):
			politicsWords.extend(a)
		elif (topic == SOCCER):
			soccerWords.extend(a)
		elif (topic == WORLDNEWS):
			worldNewsWords.extend(a)


	hockeyFreq = FreqDist(word.lower() for word in hockeyWords)
	moviesFreq = FreqDist(word.lower() for word in moviesWords)
	nbaFreq = FreqDist(word.lower() for word in nbaWords)
	newsFreq = FreqDist(word.lower() for word in newsWords)
	nflFreq = FreqDist(word.lower() for word in nflWords)
	politicsFreq = FreqDist(word.lower() for word in politicsWords)
	soccerFreq = FreqDist(word.lower() for word in soccerWords)
	worldNewsFreq = FreqDist(word.lower() for word in worldNewsWords)

	commonHockey = [ x[0] for x in hockeyFreq.most_common(frequencyNumber) ]
	commonMovies = [ x[0] for x in moviesFreq.most_common(frequencyNumber) ]
	commonNba = [ x[0] for x in nbaFreq.most_common(frequencyNumber) ]
	commonNews = [ x[0] for x in newsFreq.most_common(frequencyNumber) ]
	commonNfl = [ x[0] for x in nflFreq.most_common(frequencyNumber) ]
	commonPolitics = [ x[0] for x in politicsFreq.most_common(frequencyNumber) ]
	commonSoccer = [ x[0] for x in soccerFreq.most_common(frequencyNumber) ]
	commonWorldNews = [ x[0] for x in worldNewsFreq.most_common(frequencyNumber) ]

	mostCommonWords = []

	mostCommonWords.extend(commonHockey)
	mostCommonWords.extend(commonMovies)
	mostCommonWords.extend(commonNba)
	mostCommonWords.extend(commonNews)
	mostCommonWords.extend(commonNfl)
	mostCommonWords.extend(commonPolitics)
	mostCommonWords.extend(commonSoccer)
	mostCommonWords.extend(commonWorldNews)

	# print len(mostCommonWords)

	# print len(list(set(mostCommonWords)))

	parsedInput = parseInput(data, mostCommonWords)

	decisionTree(parsedInput, result)

	# print parsedInput

main()