#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import argparse
import pandas as pd 
from matplotlib import pyplot as plt

from operator import itemgetter


class Model(object):
    """
     Polynomial Regression.
    """
    def __init__(self):
        self.weights = None
        self.power = None

    def fit(self, X, y):
        """
        Fits the polynomial regression model to the training data.

        Arguments
        ----------
        X: transformed matrix - suitable for polynomial regression
        y: response variable vector for n examples
        k: polynomial degree
        """
        self.weights = np.dot((np.linalg.inv(np.dot(X.T, X))), np.dot(X.T, y))

    def predict(self, X):
        """
        Predicts the dependent variable of new data using the model.

        Arguments
        ----------
        X: transformed matrix based on the order of polynomial

        Returns
        ----------
        response variable vector for n examples
        """
        return np.dot(X, self.weights)

    def rmse(self, X, y):
        """
        Returns the RMSE(Root Mean Squared Error) when the model is validated.
        
        Arguments
        ----------
        X: transformed matrix based on the order of polynomial
        y: response variable vector for n examples
        
        Returns
        ----------
        RMSE when model is used to predict y
        """
        predicted_values = self.predict(X)
        return np.sqrt(np.sum((predicted_values - y) ** 2) / X.shape[0])


def transform_data(x, order):
    """
    Tranforms the input data based on the order of the polynomial

    Arguments
    ----------
    X: raw input matrix

    Returns
    ----------
    transformed matrix - suitable for polynomial regression (output is dependent on the order of polynomial)
    """

    # add a column of ones for bias
    x = np.c_[np.ones(x.shape[0]), x]

    for i in range(2, order+1):
        x = np.c_[x, np.power(x[:, 1], i)]
    return x

#run command:
#python poly.py --data=data/poly_reg_data.csv


if __name__ == '__main__':

    #Read command line arguments
    parser = argparse.ArgumentParser(description='Fit a Polynomial Regression Model')
    parser.add_argument('--data', required=True, help='The file which contains the dataset.')
                        
    args = parser.parse_args()

    input_data = pd.read_csv(args.data)
    input_data = input_data.sample(frac=1, random_state=13)

    n = len(input_data['y'])
    n_train = 25
    n_val = n - n_train

    x = input_data['x']
    x_train = x[:n_train][:,None]
    x_val = x[n_train:][:,None]

    y= input_data['y']
    y_train = y[:n_train][:,None]
    y_val = y[n_train:][:,None]

    poly_reg = Model()

    k = []
    for i in range(1, 11):
        k.append(i)

    #plot validation rmse versus k
    validation_errors = []
    training_errors = []
    for order in k:
        x_train_poly = transform_data(x_train, order)
        poly_reg.fit(x_train_poly, y_train)
        training_errors.append(poly_reg.rmse(x_train_poly, y_train))

        x_val_poly = transform_data(x_val, order)
        validation_errors.append(poly_reg.rmse(x_val_poly, y_val))


    validation_errors_numpy_array = np.array(validation_errors)
    min_val_index = np.where(validation_errors_numpy_array == np.amin(validation_errors_numpy_array))[0][0]
    print("Min Value of RMSE on Validation Set is - " + str(validation_errors_numpy_array[min_val_index]) + " and is obtained for polynomial of order - " + str(k[min_val_index]))

    plt.plot(k, validation_errors, marker='o')
    plt.title("Polynomial Order (k) vs Validation Errors")
    plt.xlabel("Polynomial Order")
    plt.ylabel("Validation Errors")
    plt.savefig('validation_errors.png')
    plt.show()

    #plot training rmse versus k
    training_errors_numpy_array = np.array(training_errors)
    min_val_index = np.where(training_errors_numpy_array == np.amin(training_errors_numpy_array))[0][0]
    print("Min Value of RMSE on Training Set is - " + str(
        training_errors_numpy_array[min_val_index]) + " and is obtained for polynomial of order - " + str(
        k[min_val_index]))

    plt.plot(k, training_errors, marker='o')
    plt.title("Polynomial Order (k) vs Training Errors")
    plt.xlabel("Polynomial Order")
    plt.ylabel("Training Errors")
    plt.savefig('training_errors.png')
    plt.show()

    #plot fitted polynomial curve versus k as well as the scattered training data points
    for order in [1, 3, 5, 10]:
        x_train_poly = transform_data(x_train, order)
        poly_reg.fit(x_train_poly, y_train)
        predicted_val = poly_reg.predict(x_train_poly)

        plt.scatter(x_train, y_train)
        sort_axis = itemgetter(0)
        sorted_zip = sorted(zip(x_train, predicted_val), key=sort_axis)
        x, predicted_val = zip(*sorted_zip)
        plt.plot(x, predicted_val, color='m')
        plt.title("Fitting Polynomial of Order " + str(order))
        plt.xlabel("X - Input Features")
        plt.ylabel("y - Predicted Values")
        plt.savefig("polynomial_regression_order_"+str(order)+".png")
        plt.show()

