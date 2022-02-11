import numpy as np
from scipy.spatial.distance import cdist

class Silhouette:
    def __init__(self, metric: str = "euclidean"):
        """
        inputs:
            metric: str
                the name of the distance metric to use
        """
        self.metric = metric

    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        calculates the silhouette score for each of the observations

        inputs:
            X: np.ndarray
                A 2D matrix where the rows are observations and columns are features. 

            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`

        outputs:
            np.ndarray
                a 1D array with the silhouette scores for each of the observations in `X`
        """
        # calculate distances between all points
        dists = cdist(X, X, metric=self.metric)
        
        # count observations
        num_obs = np.shape(y)[0]
        
        # create empty 1d array for scores
        scores = np.empty(shape=(num_obs,1))
        
        # calculate score for each observation
        for obs in range(num_obs): 
            
            # determine cluster membership and calculate mean in cluster distance from obs
            curr_cluster = y[obs]
            in_cluster = np.where(y==curr_cluster)[0] # indices of observations in the cluster
            out_of_cluster = np.where(y!=curr_cluster)[0] # indices of observations outside the cluster 
            a = np.mean([dists[obs, other] for other in in_cluster if not obs==other])
            
            # calculate mean nearest cluster distance from obs
            b = np.min([np.mean([dists[obs, other] for other in out_of_cluster]) for cluster in set(y) if not cluster==curr_cluster]) ## need to make nearest cluster
            
            # calculate silhouette distance
            scores[obs] = (b-a)/max(b,a)
        
        return scores