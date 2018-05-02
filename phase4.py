import glob, os
import pysrt
import re
from gensim import corpora, models, similarities
from collections import defaultdict
from pprint import pprint
import nltk.tokenize
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import nltk.data
nltk.download('punkt')
nltk.download('stopwords')    # ONLY NEED TO RUN THIS ONCE! UNCOMMENT FOR FIRST RUN
import json
from itertools import accumulate

""" Topic modeling: given a set of keyword clusters and their word distributions: use plsa/lda to 
do topic modeling with number of topics = number of clusters = C . Segment the original text 
into sentences ( paragraphs ?), then match keyword clusters to the N most probable segments.
"""

def topicModeling(clusteredKeywords, original_txt):
	
	p_stemmer = PorterStemmer()
	tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
	tokenizer2 = nltk.tokenize.RegexpTokenizer(r'\w+')

	fp = open(original_txt)
	data = fp.read()
	sentences = [s.lower().replace('-\n','').replace('\n', ' ') for s in tokenizer.tokenize(data) if len(s)>3]
	stemmedSent = {} 
	
	for s in sentences:
		stemmed = p_stemmer.stem(s)
		stemmedNoStop = [w for w in tokenizer2.tokenize(stemmed) if w not in stopwords.words('english')]
		stemmedSent[s] = stemmedNoStop

	# print('\n-----\n'.join(stemmedSent.keys()))
	print(stemmedSent)

	docs = stemmedSent.values()
	print(len(docs))

	
	dictionary = corpora.Dictionary(docs)
	print(dictionary)
	corpus = [dictionary.doc2bow(d) for d in docs]
	print("Loaded corpus...")
	
	with open(clusteredKeywords, 'r') as fp:
		clusters = json.loads(fp.read())
	C = len(clusters)

	ldamodel = models.ldamodel.LdaModel(corpus, num_topics=C, id2word = dictionary, passes=20)
	pprint(ldamodel.print_topics(num_topics=C))

	return

M = topicModeling("clusters.json", "./original_txt/A Brief Introduction to Neural Networks.txt")