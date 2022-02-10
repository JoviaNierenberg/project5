import numpy as np
from scipy.spatial.distance import cdist

class KMeans:
    def __init__(
            self,
            k: int,
            metric: str = "euclidean",
            tol: float = 1e-6,
            max_iter: int = 100):
        """
        inputs:
            k: int
                the number of centroids to use in cluster fitting
            metric: str
                the name of the distance metric to use
            tol: float
                the minimum error tolerance from previous error during optimization to quit the model fit
            max_iter: int
                the maximum number of iterations before quitting model fit
        """
        self.k = k
        self.metric = metric
        self.tol = tol
        self.max_iter = max_iter
    
    def fit(self, mat: np.ndarray):
        """
        fits the kmeans algorithm onto a provided 2D matrix

        inputs: 
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """
        
        # create random centers
        centers = np.random.uniform(np.min(mat), np.max(mat), size=(self.k,2))
        
        # set baseline values
        i=1
        mse=1
        mse_diff=1
        
        # iteratively select new centers
        while i<self.max_iter and mse_diff>self.tol:  
            # assign points to each cluster based on their distance to the centers
            dist_mat = cdist(mat, centers, metric=self.metric) # calculate distances from each point to each center, using supplied metric
            calc_labels = np.argmin(dist_mat, axis=1) # find index of minmum value in each row
            
            # calculate error - calculate squared distance from each point to its corresponding centroid, 
            # using the supplied metric, then take the mean of those values
            old_mse = mse
            mse = np.mean(np.square(np.choose(calc_labels, dist_mat.T))) # numpy.choose which constructs an array from an index array - https://stackoverflow.com/questions/17074422/select-one-element-in-each-row-of-a-numpy-array-by-column-indices
            mse_diff = abs(old_mse-mse)
            
            # update centers based on cluster membership of datapoints
            centers = np.array([mat[calc_labels==j].mean(0) for j in range(self.k)])
            
            # increment
            i+=1
        
        # store centers, labels, and final mse
        self.centers = centers
        self.labels = calc_labels
        self.mse = mse
        
    def predict(self, mat: np.ndarray) -> np.ndarray:
        """
        predicts the cluster labels for a provided 2D matrix

        inputs: 
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features

        outputs:
            np.ndarray
                a 1D array with the cluster label for each of the observations in `mat`
        """
        return self.labels

    def get_error(self) -> float:
        """
        returns the final squared-mean error of the fit model

        outputs:
            float
                the squared-mean error of the fit model
        """
        return self.mse

    def get_centroids(self) -> np.ndarray:
        """
        returns the centroid locations of the fit model

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """
        return self.centers