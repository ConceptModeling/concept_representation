#!/usr/bin/python2

import sys
if len(sys.argv) < 2:
  print("No input file specified")
  sys.exit()

from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error
import json
import numpy as np

# NCLUSTERS = 100

""" python cluster_neighborhoods.py "computer_vision_distributions.json" "computer_vision_clusters.json" 100
"""

""" Used sklearn's Kmeans clustering to cluster concept keywords, K = number of clusters,
T = minimum similarity threshold between two vectors in the same cluster.
"""
def clusters(keywordVecDict, K):
  # Number of clusters
  kmeans = KMeans(n_clusters=K)
  # Fitting the input data
  kmeans = kmeans.fit(np.array(keywordVecDict.values()))
  # Getting the cluster labels
  labels = kmeans.predict(np.array(keywordVecDict.values()))
  # print(labels)
  # Centroid values
  centroids = kmeans.cluster_centers_
  # clustersOfKeywords = []

  # for v in centroids:
  #   clustersOfKeywords = []
  #   for k in keywordVecDict:
  #     if mean_squared_error(keywordVecDict[k], v)< T:
  #       clustersOfKeywords[v].append(k)

  kclusters = [[] for _ in range(0,K)]
  for i,k in enumerate(keywordVecDict.keys()):
    # print("{0} : {1}".format(k,labels[i]))
    kclusters[labels[i]].append(k)

  for i in range(0,K):
    print("Cluster {0}".format(i))
    for item in kclusters[i]:
      print("  {0}".format(item.encode("utf-8")))

  with open(sys.argv[2],'w') as fp:
    json.dump(kclusters, fp)
  return kclusters

with open(sys.argv[1], 'r') as fp:
    data = json.loads(fp.read())
    # print(data)

C = clusters(data, int(sys.argv[3]))
