#!/usr/bin/python

import re
from gensim.models import TfidfModel
from gensim.corpora import Dictionary
import nltk.tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import numpy as np
import codecs, json
import sys

""" Returns a dictionary of {keyword: neighborhood word distribution}
M is the range of the word neighborhood: M words before the keyword + M words after.
"""
def neighborhood(input_file, M):
	splitOn = " "
	array = []
	with open(input_file, "r") as lines:
		for l in lines:
			tokens = l.split('\t')
			if tokens[0] not in ".,!@#$%^&*()[]{};:/?~`=+<>":
				array.append((tokens[0], tokens[1].strip('\n')))

	#print(array)
	nhoods = {}
	keyword = ""
	n = ""
	i=0
	while i < len(array):
		pair = array[i]
		count = i
		while pair[1] == "B" or pair[1]== "I":
			keyword += pair[0] + " "
			count += 1
			pair = array[count]
		if len(keyword) > 0:
			#print(keyword)
			n = " ".join(str(x) for x in [" ".join(p[0] for p in array[max(0,i-M):min(count+M, len(array))])])
			#n = [p[0] for p in array[max(0,i-M):min(count+M, len(array))]]
			#print("NEW: ", len(n), count-i)
			nhoods[keyword] = n


		keyword = ""
		n = ""
		i = count+1

	# print(nhoods)
	return nhoods

""" Normalizes the input vector to a unit vector by
dividing by the sum of input vector's elements
"""
def normalize(vector):
	s = np.sum(vector)
	if s <= 0:
		print(s)
	norm = np.divide(vector, s)
	return norm

""" Returns a list of sets of keyword and phrases
Each set in the list describes one concept. Use KL-divergence with threshold T to
make decision whether to group two words or not.
"""
def neighborhoodTfidfVectors(original_text, input_neighborhoods):
	tokenizer = nltk.tokenize.RegexpTokenizer(r'\w+')
	p_stemmer = PorterStemmer()
	fobj = open(original_text, "r")
	raw = fobj.read()
	tokens = tokenizer.tokenize(raw)
	# stopped_tokens = [i for i in tokens if not i in stopwords.words('english') and len(i)>1]    # REMOVES ENGLISH STOP WORDS AND WORDS WITH ONLY ONE CHAR, LIKE '1', '0' AND 'X'
	fullText = [p_stemmer.stem(i) for i in tokens]

	tokdict = {}
	# nhoods = []
	for k in input_neighborhoods:
		doc = input_neighborhoods[k]
		t = tokenizer.tokenize(doc)
		stemmed = [p_stemmer.stem(i) for i in t]
		# nhoods.append(stemmed)
		tokdict[k] = stemmed
	# print(nhoods)

	dct = Dictionary([fullText])
	# dct = Dictionary(nhoods)
	# print(dct)

	cordict = {}
	# corpus = [dct.doc2bow(hood) for hood in nhoods]
	corpus = []
	for h in tokdict:
		corpus.append(dct.doc2bow(tokdict[h]))
		cordict[h] = dct.doc2bow(tokdict[h])
	# print(corpus)
	model = TfidfModel(corpus, normalize=True)
	vectordict = {}
	for h in cordict:
		vectordict[h] = model[cordict[h]]
		# vectors.append(vector)
	# print(vectordict)

	voclen = len(dct)
	wordDistDict = {}
	for keyword in vectordict:
		wordDist = np.zeros(voclen)
		v = vectordict[keyword]
		for t in v:
			wordDist[t[0]] = t[1]
		wordDistDict[keyword.strip()] = normalize(wordDist).tolist()
	# print(np.sum(wordDistDict['perceptron']))
	# print(wordDistDict)
	return wordDistDict


if (__name__ == "__main__"):
    nhoods = neighborhood(sys.argv[1],int(sys.argv[2]))
    distributions = neighborhoodTfidfVectors(sys.argv[3], nhoods)
    json.dump(distributions, codecs.open(sys.argv[4], 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4) ### this saves the array in .json format

# nhoods = neighborhood("./tagged_txt/A Brief Introduction to Neural Networks.txt", 10)
# distributions = neighborhoodTfidfVectors("./original_txt/A Brief Introduction to Neural Networks.txt", nhoods)

# json.dump(distributions, codecs.open('distributions.json', 'w', encoding='utf-8'), separators=(',', ':'), sort_keys=True, indent=4) ### this saves the array in .json format


# with open('distributions.json','w') as fp:
# 	json.dump(distributions, fp)

# with open('distributions.json', 'r') as fp:
#     data = json.loads(fp.read())

# print(data)





