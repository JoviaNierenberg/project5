# Importing Dependencies
import pytest
import numpy as np
from scipy.spatial.distance import cdist
from cluster import Silhouette, make_clusters

# write your silhouette score unit tests here
def test_silhouette():
	"""
	Tests that silhouette scores are between -1 and 1
	"""
	clusters, labels = make_clusters(k=4, scale=1)
	sil=Silhouette()
	sil_scores = sil.score(clusters, labels)
	assert np.max(sil_scores)<1 and np.min(sil_scores)>-1

