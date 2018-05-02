from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
import json
import numpy as np 




""" Used sklearn's Kmeans clustering to cluster concept keywords, K = number of clusters,
T = minimum similarity threshold between two vectors in the same cluster.
"""
def clusters(keywordVecDict, K, T):
	# Number of clusters
	kmeans = KMeans(n_clusters=K)
	# Fitting the input data
	kmeans = kmeans.fit(np.array(keywordVecDict.values()))
	# Getting the cluster labels
	labels = kmeans.predict(np.array(keywordVecDict.values()))
	print labels
	# Centroid values
	centroids = kmeans.cluster_centers_
	clustersOfKeywords = []

	for v in centroids:
		clustersOfKeywords = []
		for k in keywordVecDict:
			if mean_squared_error(keywordVecDict[k], v)< T:
				clustersOfKeywords[v].append(k)

	return centroids

with open('distributions.json', 'r') as fp:
    data = json.loads(fp.read())

# print data.keys()

C = clusters(data, 10, 0.7)