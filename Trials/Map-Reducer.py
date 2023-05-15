from mrjob.job import MRJob
from mrjob.protocol import JSONValueProtocol
from sklearn.linear_model import LinearRegression
import numpy as np
class MRLinearRegression(MRJob):


    def mapper_init(self):
        # Load the training data from a CSV file
        data = np.loadtxt("/media/menna/New Volume/Desktop/college/4th Year/2nd semester/Big Data/new.csv", delimiter=",", skiprows=0)
        self.X_train = np.array(data[:, :-1])
        self.y_train = np.array(data[:, -1])
        # Initialize the linear regression model
        self.lr = LinearRegression(fit_intercept=False)
        # Train the model on the training data
        self.lr.fit(self.X_train, self.y_train)

    def mapper(self, _, line):
        data_point = np.array([float(x) for x in line.split(',')])

        # Extract the features from the input data point
        x = data_point[0:13]
        # Extract the target variable from the input data point
        y = data_point[13]
        # Make a prediction on the input data point using the trained model
        x = x.reshape(1, -1)
        y_pred = self.lr.predict(x)
        # Emit the predicted value as the output key and the input value as the output value
        yield str(y_pred[0]), {'features': x.tolist(), 'target': y}

    def reducer(self, key, values):
        # Convert the input values to a list
        values_list = list(values)
        # Compute the mean of the target variable in the input values
        mean_target = np.mean([v['target'] for v in values_list])
        # Compute the mean squared error between the predicted value and the mean of the target variable
        mse = np.mean([(float(key) - v['target'])**2 for v in values_list])
        # Emit the mean target variable and the mean squared error as the output value
        yield key, {'mean_target': mean_target, 'mse': mse}

if __name__ == '__main__':
    MRLinearRegression.run()