from sklearn.metrics import mutual_info_score
import numpy as np
from operator import itemgetter
import codecs, json
import sys
from pprint import pprint

"""
python find_coverage.py "cluster_topics.json" "sentence_topics.json" 5
"""

""" Given the probability distributions of topics for each cluster and for each sentence in the 
original text, return for each cluster the N sentences with the lowest kl-divergence to the 
cluster's topic distribution.
"""
def topNsentences(cluster_topics, sentence_topics, N):

	with open(cluster_topics, 'r') as f1:
		C_topics = json.loads(f1.read())

	with open(sentence_topics, 'r') as f2:
		S_topics = json.loads(f2.read())

	# pprint(C_topics)
	# pprint(S_topics)

	numClustr = len(C_topics)


	# MATRIX OF MUTUAL INFORMATION SCORES
	scores = np.zeros((len(C_topics), len(S_topics)))
	concept_lookup = {}
	sent_lookup = {}
	for i in range(len(C_topics.keys())):
		concept = C_topics.keys()[i]
		ct = [ t[1] for t in C_topics[concept]]
		concept_lookup[i] = concept
		for j in range(len(S_topics.keys())):
			sentence = S_topics.keys()[j]
			st = [ t[1] for t in S_topics[sentence]]
			sent_lookup[j] = sentence
			scores[i][j] = mutual_info_score(ct, st)

	pprint(scores)

	(c,s) = np.shape(scores)
	coverage_dict = {}
	for i in range(c):
		con = concept_lookup[i]
		simil = scores[i]
		topInx = simil.argsort()[-N:][::-1]
		coverage_dict[con] = []

		for n in range(len(topInx)):
			sent = sent_lookup[topInx[n]]
			coverage_dict[con].append(sent)
	
	pprint(coverage_dict)
	json.dump(coverage_dict, codecs.open('coverage_dict.json', 'w', encoding='utf-8'), separators=(',', ':'), indent=4) ### this saves the array in .json format
	
	return

# coverage = topNsentences("cluster_topics.json", "sentence_topics.json", 5)
coverage = topNsentences(sys.argv[1], sys.argv[2], int(sys.argv[3]))


