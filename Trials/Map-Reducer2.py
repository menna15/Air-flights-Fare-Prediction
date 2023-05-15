from mrjob.job import MRJob
from mrjob.step import MRStep
import numpy as np

class LinearRegression(MRJob):
    
        
    def mapper(self, _, line):
        """
        The mapper reads in each line of the input file, splits it by commas,
        and yields a tuple of key-value pairs, where the key is None and the
        value is a tuple containing the label and the features.
        """

        parts = line.strip().split(',')
        label = float(parts[13])
        features = np.array([float(x) for x in parts[0:13]])
        yield None, (label, features.tolist())

    
    def reducer_init(self):
        """
        The reducer initializes the weights randomly and sets the learning rate
        and number of iterations for gradient descent.
        """
        self.learning_rate = 0.01
        self.num_iterations = 10
        self.weights = np.random.randn(13)
    
    def reducer(self, _, values):
        """
        The reducer reads in all the key-value pairs from the mapper and combiner and
        trains a linear regression model using batch gradient descent.
        """
        X = []
        y = []
        for label, features in values:
            X.append(features)
            y.append(label)
        X = np.array(X)
        y = np.array(y)
        for i in range(self.num_iterations):
            y_pred = np.dot(X, self.weights)
            error = y_pred - y
            gradient = np.dot(X.T, error) / len(y)
            self.weights -= self.learning_rate * gradient
        yield None, (self.weights).tolist()

    def reducer_final(self, _, values):
        """
        The reducer final method accumulates the weights and yields a single key-value pair
        with the final weights.
        """
        weights_accumulated = np.zeros(13)
        for weights in values:
            weights_accumulated += weights
        yield None, weights_accumulated.tolist()
    
    def steps(self):
        """
        The steps method defines the MapReduce job with two steps: mapper/combiner and reducer.
        """
        return [
            MRStep(mapper=self.mapper,
                   reducer_init=self.reducer_init,
                   reducer=self.reducer),
            MRStep(reducer_final=self.reducer_final)
        ]

if __name__ == '__main__':
    LinearRegression.run()