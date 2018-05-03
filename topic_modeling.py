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
from operator import itemgetter
import codecs, json
import sys

""" python3 topic_modeling.py "computer_vision_clusters.json" "./original_txt/Computer Vision - Algorithms and Applications.txt" 'comp_vis_cluster_topics.json' 'comp_vis_sentence_topics.json'
"""

""" Topic modeling: given a set of keyword clusters and their word distributions: use plsa/lda to 
do topic modeling with number of topics = number of clusters = C . Segment the original text 
into sentences ( paragraphs ?), then match keyword clusters to the N most probable segments.
"""

def topicModeling(clusteredKeywords, original_txt):
	print("Begin...")
	p_stemmer = PorterStemmer()
	tokenizer1 = nltk.data.load('tokenizers/punkt/english.pickle')
	# tokenizer1 = nltk.tokenize.TextTilingTokenizer()
	tokenizer2 = nltk.tokenize.RegexpTokenizer(r'\w+')
	print("Loaded tokenizers...")

	fp = open(original_txt)
	data = fp.read()
	print("Read original_txt...")
	tokens_1 = tokenizer1.tokenize(data)
	print(len(tokens_1))
	sentences = [s.lower().replace('-\n','').replace('\n', ' ') for s in tokens_1 if len(s)>3]
	# pprint(sentences)
	print("Finished sentences...")
	
	stemmedSent = {} 
	for s in sentences:
		stemmed = p_stemmer.stem(s)
		stemmedNoStop = [w for w in tokenizer2.tokenize(stemmed) if w not in stopwords.words('english')]
		stemmedSent[s] = stemmedNoStop

	# print('\n-----\n'.join(stemmedSent.keys()))
	# print(stemmedSent)
	print("Finished stemmedSent...")

	
	parag_len = 5
	paragraphs = {}
	counter = 0
	current_key = []
	current_value = []
	for s in stemmedSent:
		if counter<5:
			current_key.append(s)
			current_value.extend(stemmedSent[s])
			counter+=1
		else:
			k = ' '.join(current_key)
			paragraphs[k] = current_value
			counter=0
			current_key = []
			current_value = []

	print(paragraphs)
	print("Finished paragraphs...")
	# docs = stemmedSent.values()
	docs = paragraphs.values()
	print(len(docs))

	
	dictionary = corpora.Dictionary(docs)
	print(dictionary)
	corpus = [dictionary.doc2bow(d) for d in docs]
	print("Loaded corpus...")
	
	with open(clusteredKeywords, 'r') as fp:
		clusters = json.loads(fp.read())
	C = len(clusters)

	ldamodel = models.ldamodel.LdaModel(corpus, num_topics=C, minimum_probability=0.0, id2word = dictionary, passes=20)
	# pprint(ldamodel.print_topics(num_topics=C))
	print("Finished topic modeling...")

	# CLUSTER TOPICS
	cluster_topics = {}
	for c in clusters:
		current = ' '.join(c)
		doc_bow = dictionary.doc2bow(c)
		topic = ldamodel[doc_bow]
		# topic.sort(key=itemgetter(1), reverse=True)
		topic.sort(key=itemgetter(0))
		cluster_topics[current] = topic
	# pprint(cluster_topics)

	# TOPICS PER SENTENCE
	sentence_topics = {}
	# for d in stemmedSent.keys():
	for d in paragraphs:
		# current = ' '.join(d)
		current = d 
		doc_bow = dictionary.doc2bow(paragraphs[d])
		topic = ldamodel[doc_bow]
		# topic.sort(key=itemgetter(1), reverse=True)
		topic.sort(key=itemgetter(0))
		sentence_topics[current] = topic
	# pprint(sentence_topics)

	# 'cluster_topics.json'
	# 'sentence_topics.json'
	print("Writing to json...")
	json.dump(cluster_topics, codecs.open( sys.argv[3], 'w', encoding='utf-8'), separators=(',', ':'), indent=4) ### this saves the array in .json format
	json.dump(sentence_topics, codecs.open(sys.argv[4], 'w', encoding='utf-8'), separators=(',', ':'), indent=4) ### this saves the array in .json format
	print("Done!")
	return



# M = topicModeling("clusters.json", "./original_txt/A Brief Introduction to Neural Networks.txt")

M = topicModeling(sys.argv[1], sys.argv[2])



