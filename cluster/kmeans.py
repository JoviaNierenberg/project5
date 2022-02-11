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

        # model should not run if k<=0
        if self.k<=0:
            raise ValueError("k must be greater than zero")
    
    def fit(self, mat: np.ndarray):
        """
        fits the kmeans algorithm onto a provided 2D matrix

        inputs: 
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """
        
        # model should not run if k > number of observations
        if self.k > np.shape(mat)[0]:
            raise ValueError("k must be less than the number of observations in the array")

        # create random centers
        self.centers = np.random.uniform(np.min(mat), np.max(mat), size=(self.k,2))
        
        # set baseline values
        i=1
        mse=1
        mse_diff=1
        self.mat = mat # necessary to access mat in get_centroids
        
        # iteratively select new centers
        while i<self.max_iter and mse_diff>self.tol:  
            # assign points to each cluster based on their distance to the centers
            self.dist_mat = self._calc_distances(mat, self.centers)
            self.labels = self.predict(mat)
            
            # calculate error - calculate squared distance from each point to its corresponding centroid, 
            # using the supplied metric, then take the mean of those values
            old_mse = mse
            mse = self.get_error()
            mse_diff = abs(old_mse-mse)
            
            # update centers based on cluster membership of datapoints
            self.centers = self.get_centroids()
            
            # increment
            i+=1

        
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
        return np.argmin(self.dist_mat, axis=1) # find index of minmum value in each row

    def get_error(self) -> float:
        """
        returns the final squared-mean error of the fit model

        outputs:
            float
                the squared-mean error of the fit model
        """
        return np.mean(np.square(np.choose(self.labels, self.dist_mat.T))) # numpy.choose which constructs an array from an index array - https://stackoverflow.com/questions/17074422/select-one-element-in-each-row-of-a-numpy-array-by-column-indices

    def get_centroids(self) -> np.ndarray:
        """
        returns the centroid locations of the fit model

        outputs:
            np.ndarray
                a `k x m` 2D matrix representing the cluster centroids of the fit model
        """
        return np.array([self.mat[self.labels==j].mean(0) for j in range(self.k)])

    def _calc_distances(self, mat: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """
        calculates the distance from each point to each center
        """
        return cdist(mat, centers, metric=self.metric)

