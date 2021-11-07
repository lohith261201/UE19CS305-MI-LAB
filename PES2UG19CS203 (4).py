import numpy as np
from math import *
from decimal import Decimal


class KNN:
    """
    K Nearest Neighbours model
    Args:
        k_neigh: Number of neighbours to take for prediction
        weighted: Boolean flag to indicate if the nieghbours contribution
                  is weighted as an inverse of the distance measure
        p: Parameter of Minkowski distance
    """

    def __init__(self, k_neigh, weighted=False, p=2):
        
        dummy1= k_neigh
        dummy2= p
        self.k_neigh = dummy1
        self.p = dummy2
        self.weighted = weighted

    def fit(self, data, target):
        """
        Fit the model to the training dataset.
        Args:
            data: M x D Matrix( M data points with D attributes each)(float)
            target: Vector of length M (Target class for all the data points as int)
        Returns:
            The object itself
        """
        alias = data
        self.data = alias
        self.target = target.astype(np.int64)

        return self

    def Proot(self, value, root):
        tmper=round(Decimal(value) ** Decimal(1 / float(root)), 3)
        return tmper

    def minkowski_distance(self, x, y, p_value):
        returnval=float(self.Proot(sum(pow(abs(n-m), p_value)
                                        for n, m in zip(x, y)), p_value))
        return returnval

    def find_distance(self, x):
        """
        Find the Minkowski distance to all the points in the train dataset x
        Args:
            x: N x D Matrix (N inputs with D attributes each)(float)
        Returns:
            Distance between each input to every data point in the train dataset
            (N x M) Matrix (N Number of inputs, M number of samples in the train dataset)(float)
        """
        # TODO
        DIST = []
        dummy=x.shape[0]
        for k in range(dummy):
            temp = []
            a = x[k]
            d=self.data
            c=d.shape[0]
            for j in range(c):
                b = d[j]
                temp.append(self.minkowski_distance(a, b , self.p))
            DIST.append(temp)

        return DIST

    def k_neighbours(self, x):
        """
        Find K nearest neighbours of each point in train dataset x
        Note that the point itself is not to be included in the set of k Nearest Neighbours
        Args:
            x: N x D Matrix( N inputs with D attributes each)(float)
        Returns:
            k nearest neighbours as a list of (neigh_dists, idx_of_neigh)
            neigh_dists -> N x k Matrix(float) - Dist of all input points to its k closest neighbours.
            idx_of_neigh -> N x k Matrix(int) - The (row index in the dataset) of the k closest neighbours of each input

            Note that each row of both neigh_dists and idx_of_neigh must be SORTED in increasing order of distance
        """
        # TODO
        KNN = [[], []]
        DIST = self.find_distance(x)
        a=len(DIST)
        for i in range(a):
            indexes = [i for i in range(self.data.shape[0])]
            e = list(list(zip(*list(sorted(zip(DIST[i], indexes)))))[1])
            #d = list(list(zip(*list(sorted(zip(DIST[i], indexes)))))[0])
            KNN[0].append(list(list(zip(*list(sorted(zip(DIST[i], indexes)))))[0])[0:self.k_neigh])
            KNN[1].append(e[0:self.k_neigh])

        return KNN
    def predict(self, x):
        """
        Predict the target value of the inputs.
        Args:
            x: N x D Matrix( N inputs with D attributes each)(float)
        Returns:
            pred: Vector of length N (Predicted target value for each input)(int)
        """
        # TODO
        dummy=self.k_neighbours(x)
        targetvec= []
        indices = dummy[1]
        a=len(indices)
        for i in range(a):
            seteg = {}
            dummy2=len(indices[i])
            for j in range(dummy2):
                if self.target[indices[i][j]] not in seteg:
                    seteg[self.target[indices[i][j]]] = 1
                else:
                    seteg[self.target[indices[i][j]]] += 1
            max_K = None
            max_F = 0
            for i in range(min(seteg), max(seteg)+1):
                if seteg[i] <= max_F:
                    pass
                else:    
                    max_F = seteg[i]
                    max_K = i
            targetvec.append(max_K)
        return targetvec

    def evaluate(self, x, y):
        """
        Evaluate Model on test data using 
            classification: accuracy metric
        Args:
            x: Test data (N x D) matrix(float)
            y: True target of test data(int)
        Returns:
            accuracy : (float.)
        """
        correctcount = 0
        a=self.predict
        result = a(x)
        b=len(y)
        for k in range(b):
            if y[k] != result[k]:
                pass
            else:    
                correctcount += 1
        accuracy = correctcount / float(b) * 100.0
        return correctcount / float(b) * 100.0
