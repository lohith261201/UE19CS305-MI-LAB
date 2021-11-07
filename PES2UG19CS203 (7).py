import numpy as np
import math
from sklearn.tree import DecisionTreeClassifier

"""
Use DecisionTreeClassifier to represent a stump.
------------------------------------------------
DecisionTreeClassifier Params:
    critereon -> entropy
    max_depth -> 1
    max_leaf_nodes -> 2
Use the same parameters
"""
# REFER THE INSTRUCTION PDF FOR THE FORMULA TO BE USED 

def sign(x):
	return abs(x)/x if x!=0 else 1

class AdaBoost:

    """
    AdaBoost Model Class
    Args:
        n_stumps: Number of stumps (int.)
    """

    def __init__(self, n_stumps=20):

        self.n_stumps = n_stumps
        self.stumps = []

    def fit(self, X, y):
        """
        Fitting the adaboost model
        Args:
            X: M x D Matrix(M data points with D attributes each)(numpy float)
            y: M Vector(Class target for all the data points as int.)
        Returns:
            the object itself
        """
        self.alphas = []

        sample_weights = np.ones_like(y) / len(y)
        for _ in range(self.n_stumps):

            st = DecisionTreeClassifier(
                criterion='entropy', max_depth=1, max_leaf_nodes=2)
            st.fit(X, y, sample_weights)
            y_pred = st.predict(X)

            self.stumps.append(st)

            error = self.stump_error(y, y_pred, sample_weights=sample_weights)
            alpha = self.compute_alpha(error)
            self.alphas.append(alpha)
            sample_weights = self.update_weights(
                y, y_pred, sample_weights, alpha)

        return self

    def stump_error(self, y, y_pred, sample_weights):
        """
        Calculating the stump error
        Args:
            y: M Vector(Class target for all the data points as int.)
            y_pred: M Vector(Class target predicted for all the data points as int.)
            sample_weights: M Vector(Weight of each sample float.)
        Returns:
            The error in the stump(float.)
        """

        # TODO
        l=len(y)
        x1=0
        x2=0
        
        
        for i in range(l):
        	if(y[i]==y_pred[i]):
        		x1=x1+0
        	else:
        		x1=x1+sample_weights[i]
        
        for j in sample_weights:
        	x2+=j	
        return(x1/x2)

    def compute_alpha(self, error):
        """
        Computing alpha
        The weight the stump has in the final prediction
        Use eps = 1e-9 for numerical stabilty.
        Args:
            error:The stump error(float.)
        Returns:
            The alpha value(float.)
        """
        eps = 1e-9
        epsorig=eps
        # TODO
        alpha = (0.5) * math.log(((1 - error + epsorig) / (error + epsorig)))
        origet=alpha
        return origet

    def update_weights(self, y, y_pred, sample_weights, alpha):
        """
        Updating Weights of the samples based on error of current stump
        The weight returned is normalized
        Args:
            y: M Vector(Class target for all the data points as int.)
            y_pred: M Vector(Class target predicted for all the data points as int.)
            sample_weights: M Vector(Weight of each sample float.)
            alpha: The stump weight(float.)
        Returns:
            new_sample_weights:  M Vector(new Weight of each sample float.)
        """

        # TODO
        new_weights=[]
        l=len(y)

        
        for i in range(l):
        	 if(y[i]!=y_pred[i]):
        	 	new_weights.append(sample_weights[i] * math.exp(alpha))
        	 
        	 else:
        	 	new_weights.append(sample_weights[i] * math.exp(-alpha))
        		
        
        weSum=sum(new_weights)
        dummy=weSum
        for j in range(l):
        	new_weights[j]=new_weights[j]/weSum
        
        	
        return new_weights 
    
    def predict(self, X):
        """
        Predicting using AdaBoost model with all the decision stumps.
        Decison stump predictions are weighted.
        Args:
            X: N x D Matrix(N data points with D attributes each)(numpy float)
        Returns:
            pred: N Vector(Class target predicted for all the inputs as int.)
        """
        # TODO
        
        preds=[]
        for k in self.alphas:
        	preds.append([k * stump.predict(X) for stump in self.stumps])
        	

        ypred = np.sum(preds, axis=0)
        ypred = np.sign(y_pred)       
        ypred=y_pred[0]
        dummy=ypred
        return dummy

    def evaluate(self, X, y):
        """
        Evaluate Model on test data using
            classification: accuracy metric
        Args:
            x: Test data (N x D) matrix
            y: True target of test data
        Returns:
            accuracy : (float.)
        """
        predi = self.predict(X)
        # find correct predictions
        crct = (predi == y)
        hundred=100

        accuracy = np.mean(crct) * hundred 
        # accuracy calculation
        dummy=accuracy
        return accuracy
