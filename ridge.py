#!/usr/bin/python
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import argparse
import math

from operator import itemgetter


class Model(object):
    """
     Ridge Regression.
    """
    def __init__(self):
        self.weights = None
     
    def fit(self, X, y, alpha=0):
        """
        Fits the ridge regression model to the training data.

        Arguments
        ----------
        X: nx(d+1) matrix of n examples with d independent variables
        y: response variable vector for n examples
        alpha: regularization parameter.
        """
        # Assumption: For including the bias term in the expression, we would have already added an extra column in the design matrix

        # fit implies we have to compute the hyper-parameters associated with the model
        # compute the modified Identity matrix with (d+1) x (d+1) matrix
        d = X.shape[1] - 1

        I = np.eye(d)    # identity matrix
        I = np.c_[np.zeros( (d, 1) ), I]
        I = np.r_[np.zeros( (1, d+1) ), I]
        self.weights = np.dot((np.linalg.inv( np.dot(X.T, X) + (alpha * I))), np.dot(X.T, y))

    def predict(self, X):
        """
        Predicts the dependent variable of new data using the model.

        Arguments
        ----------
        X: nx(d+1) matrix of n examples with d covariates

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
        X: nx(d+1) matrix of n examples with d covariates
        y: response variable vector for n examples
            
        Returns
        ----------
        RMSE when model is used to predict y
        """
        # We have to note that X -> input and y -> actual-values/true-value
        # Therefore, we have to get the predicted output first
        # We will be using the predict function that we have written above for this task
        predicted_values = self.predict(X)
        return np.sqrt(np.sum((predicted_values - y) ** 2)/X.shape[0])


#run command:
#python ridge.py --X_train_set=data/Xtraining.csv --y_train_set=data/Ytraining.csv --X_val_set=data/Xvalidation.csv --y_val_set=data/Yvalidation.csv --y_test_set=data/Ytesting.csv --X_test_set=data/Xtesting.csv

if __name__ == '__main__':

    #Read command line arguments
    parser = argparse.ArgumentParser(description='Fit a Ridge Regression Model')
    parser.add_argument('--X_train_set', required=True, help='The file which contains the covariates of the training dataset.')
    parser.add_argument('--y_train_set', required=True, help='The file which contains the response of the training dataset.')
    parser.add_argument('--X_val_set', required=True, help='The file which contains the covariates of the validation dataset.')
    parser.add_argument('--y_val_set', required=True, help='The file which contains the response of the validation dataset.')
    parser.add_argument('--X_test_set', required=True, help='The file which containts the covariates of the testing dataset.')
    parser.add_argument('--y_test_set', required=True, help='The file which containts the response of the testing dataset.')
                        
    args = parser.parse_args()

    #Parse training dataset
    X_train = np.genfromtxt(args.X_train_set, delimiter=',')
    y_train = np.genfromtxt(args.y_train_set,delimiter=',')
    
    #Parse validation set
    X_val = np.genfromtxt(args.X_val_set, delimiter=',')
    y_val = np.genfromtxt(args.y_val_set, delimiter=',')
    
    #Parse testing set
    X_test = np.genfromtxt(args.X_test_set, delimiter=',')
    y_test = np.genfromtxt(args.y_test_set, delimiter=',')

    # Add a column to the design matrix to calculate the bias parameter
    X_train = np.c_[np.ones(X_train.shape[0]), X_train]
    X_val = np.c_[np.ones(X_val.shape[0]), X_val]
    X_test = np.c_[np.ones(X_test.shape[0]), X_test]

    ridge_regressor = Model()

    a = [1, 2, 3, 4, 5, 6, 7, 8, 9]
    b = [-5, -4, -3, -2, -1, 0]
    lambda_values = []  # stores all the lambda values
    for a_val in a:
        for b_val in b:
            lambda_values.append(a_val * (10 ** b_val))

    rmse_errors = []
    for lambda_val in lambda_values:
        ridge_regressor.fit(X_train, y_train, lambda_val)
        rmse_errors.append(ridge_regressor.rmse(X_val, y_val))

    sort_axis = itemgetter(0)
    sorted_zip = sorted(zip(lambda_values, rmse_errors), key=sort_axis)
    temp_lambda_values, temp_rmse_errors = zip(*sorted_zip)

    plt.semilogx(temp_lambda_values, temp_rmse_errors, marker='o')
    plt.title("RMSE Errors vs Lambda Values (semilogx values considered for lambda)")
    plt.xlabel("Lambda Values")
    plt.ylabel("RMSE Errors")
    plt.savefig('rmse_vs_lambda.png')
    plt.show()

    min_val_index = np.where(rmse_errors == np.amin(rmse_errors))[0][0]
    print("Min Value of RMSE - " + str(np.amin(rmse_errors)) +" is obtained for a lambda value - " + str(lambda_values[min_val_index]))

    xi = list(range(len(lambda_values)))
    plt.plot(xi, rmse_errors, marker = 'o')
    plt.annotate('Optimal Lambda @8th index = '+str(lambda_values[min_val_index]), xy=(xi[min_val_index], np.amin(rmse_errors)), color = 'red')
    plt.title("RMSE Errors vs Indicies of Lambda Values (Extra Visualization)")
    plt.xlabel("Indicies corresponding to Lambda Values (range 0 to 53)")
    plt.ylabel("RMSE Errors")
    plt.savefig('rmse_vs_lambda_clear.png')
    plt.show()

    #plot predicted versus real value
    optimal_lambda_val = lambda_values[min_val_index]
    ridge_regressor.fit(X_train, y_train, optimal_lambda_val)
    predicted_values = ridge_regressor.predict(X_test)

    sort_axis = itemgetter(0)
    sorted_zip = sorted(zip(y_test, predicted_values), key=sort_axis)
    temp_y_test, temp_predicted_values = zip(*sorted_zip)

    plt.scatter(temp_y_test, temp_predicted_values, marker = 'o')
    plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], 'k--', lw=4, color = 'red')
    plt.title("True vs Predicted Glucose Concentration Scatter Plot")
    plt.xlabel("True Values")
    plt.ylabel("Predicted Values")
    plt.savefig('scatter_true_predicted.png')
    plt.show()

    #plot regression coefficients
    all_beta_values = []
    for lambda_val in lambda_values:
        ridge_regressor = Model()
        ridge_regressor.fit(X_train, y_train, lambda_val)
        # pick the first 10 weights by excluding the bias term
        all_beta_values.append(ridge_regressor.weights[0:11])
    # PLOT ALL BETA VALUES vs LAMBDA VALUES
    # SemiLogx is considered for lambda values for better visualization
    # fig, axs = plt.subplots(2, 5)
    # fig.tight_layout()
    # i = 0

    for j in range(11):
        # if j == 5:
            # i += 1
        beta_j = np.array(all_beta_values)[:, j]
        sort_axis = itemgetter(0)
        sorted_zip = sorted(zip(lambda_values, beta_j), key=sort_axis)
        temp_lambda_values, temp_beta_j = zip(*sorted_zip)
        plt.semilogx(temp_lambda_values, temp_beta_j, label = 'Beta = ' + str(j))
        # axs[i, j % 5].semilogx(temp_lambda_values, temp_beta_j)
        # axs[i, j % 5].set_title("Beta " + str(j + 1))
    plt.legend(loc = 'best')
    plt.title("First 10 dimensions of beta vs shrinkage parameter lambda")
    plt.savefig('beta_coefficients.png')
    plt.show()