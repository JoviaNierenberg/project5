# Importing Dependencies
import pytest
import numpy as np
from scipy.spatial.distance import cdist
from cluster import KMeans, make_clusters

# Write your k-means unit tests here
def test_num_labels_types_is_k():
	"""
	Tests that the number of clusters produced is k
	"""
	clusters, labels = make_clusters(k=4, scale=1)
	km = KMeans(k=4)
	km.fit(clusters)
	created_labels = km.predict(clusters)
	should_be_k = np.shape(np.unique(created_labels))[0]
	assert should_be_k==4

def test_vals():
	"""
	Tests values for error and and two centroid values when k==7
	"""
	clusters, labels = make_clusters(k=4, scale=1)
	km = KMeans(k=7)
	km.fit(clusters)
	assert km.get_error() == 1.4503012126641381
	assert km.get_centroids()[0,0]==7.875495297338064
	assert km.get_centroids()[6,1]==-4.171662182273182