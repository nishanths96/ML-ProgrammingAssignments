import numpy as np
from cvxopt import matrix
from cvxopt import solvers
import time

train_data = np.loadtxt("train_data.txt")
train_label = np.loadtxt("train_label.txt")
test_data = np.loadtxt("test_data.txt")
test_label = np.loadtxt("test_label.txt")

def train_svm(train_data, train_label, C):
    """Train linear SVM (primal form)

  Argument:
    train_data: N*D matrix, each row as a sample and each column as a feature
    train_label: N*1 vector, each row as a label
    C: tradeoff parameter (on slack variable side)

  Return:
    w: feature vector (column vector)
    b: bias term
  """
    n, d = train_data.shape
    # output y's type -> float
    y = train_label.reshape(-1, 1)
    # compute y_n y_m <x_n, x_m>
    intermediate_val = y * train_data
    H = np.dot(intermediate_val, intermediate_val.T)

    # Get all the variables necessary for the convex optimization solver:
    P = matrix(H)
    # vector comprising of -1 => size: n x 1
    q = matrix(-np.ones((n, 1)))
    # have diagonal of size n x n with -1 as diagonal entries; then stack identity matrix of size n x n
    G = matrix(np.vstack((np.eye(n) * -1, np.eye(n))))
    # variable h is for RHS of the constraints; therefore, have n zeros and then, append n C's
    h = matrix(np.hstack((np.zeros(n), np.ones(n) * C)))
    # the following variables are for the other constraint between y and alpha. <y, alpha> = 0
    A = matrix(y.reshape(1, -1))
    # RHS of the constraint => 0
    b = matrix(np.zeros(1))

    # Setting solver parameters (change default to decrease tolerance)
    solvers.options['show_progress'] = False
    solvers.options['abstol'] = 1e-10
    solvers.options['reltol'] = 1e-10
    solvers.options['feastol'] = 1e-10

    # Run solver
    sol = solvers.qp(P, q, G, h, A, b)
    alphas = np.array(sol['x'])

    w = np.dot((y * alphas).T, train_data).reshape(-1, 1)
    support_vec = np.empty(0, dtype=np.bool_)
    for vec in alphas:
        #
        if (vec[0] > 1e-10) and (vec[0] < C-1e-10):
            support_vec = np.append(support_vec, True)
        else:
            support_vec = np.append(support_vec, False)

    b = y[support_vec] - np.dot(train_data[support_vec], w)
    return w, b, support_vec

def test_svm(test_data, test_label, w, b):
    """Test linear SVM

  Argument:
    test_data: M*D matrix, each row as a sample and each column as a feature
    test_label: M*1 vector, each row as a label
    w: feature vector
    b: bias term

  Return:
    test_accuracy: a float between [0, 1] representing the test accuracy
  """
    correct = 0
    test_out = test_label.reshape(-1,1) * (np.dot(test_data, w) + b)
    for n in range(len(test_out)):
        if test_out[n] > 0:
            correct += 1
    # return the test_accuracy
    return 100.0 * (correct / test_data.shape[0])

def pre_process_data(X):
    """ Mean center the data
  
  Argument: 
    X: Training Data -> M*D matrix, each row as a sample and each column as a feature
  
  Return:
    Mean of all the features
    Standard Deviation for all the features
    Mean-centered Data
  """
    n = X.shape[0]
    x_mean = np.empty(0)

    # compute mean for every column/feature
    for column in range(X.shape[1]):
        x_mean = np.append(x_mean, X[:, column].sum() / n)

    intermediate_val = X - x_mean

    # compute standard deviation for every column/feature
    x_std = np.empty(0)
    for column in range(X.shape[1]):
        x_std = np.append(x_std, np.sqrt(np.power(X[:, column] - x_mean[column], 2).sum() / (n - 1)))

    mean_centered_data = intermediate_val / x_std
    return x_mean, x_std, mean_centered_data


train_data_mean, train_data_std, mean_centered_train_data = pre_process_data(train_data)
test_data_mean, test_data_std, mean_centered_test_data = pre_process_data(test_data)
print("THIRD FEATURE: mean:", train_data_mean[2], "; standard_deviation:", train_data_std[2])
print("TENTH FEATURE: mean:", train_data_mean[9], "; standard_deviation:", train_data_std[9])
print("")

C = [4 ** -6, 4 ** -5, 4 ** -4, 4 ** -3, 4 ** -2, 4 ** -1, 4 ** 0, 4 ** 1, 4 ** 2, 4 ** 3, 4 ** 4, 4 ** 5, 4 ** 6]
# 5-fold cross validation:
fold_indicies = [list(range(0, 200)), list(range(200, 400)), list(range(400, 600)),
                 list(range(600, 800)), list(range(800, 1000))]
train_indicies = [list(range(200, 1000)), list(range(0, 200)) + list(range(400, 1000)),
                  list(range(0, 400)) + list(range(600, 1000)), list(range(0, 800))]

c_index = 0
# w, b, support_vec = train_svm(mean_centered_train_data, train_label, 4**6)
# print("w:",w.shape,
#               "b:", b.shape,
#               "Number of Support Vectors:",
#               np.unique(support_vec, return_counts = True))
#
# print("Test Accuracy: ", test_svm(mean_centered_test_data, test_label, w, b[0]))
# print("Train Accuracy: ", test_svm(mean_centered_train_data, train_label, w, b[0]))

avg_cross_val_accuracy = []
for c in C:
    print(" ---------- C =", c, "; index = ", c_index, "---------------")
    cross_val_accuracy = []
    time_taken = []
    for fold_ind, train_ind in zip(fold_indicies, train_indicies):
        val_X = mean_centered_train_data[fold_ind]
        val_y = train_label[fold_ind]
        X = mean_centered_train_data[train_ind]
        y = train_label[train_ind]

        start_time = time.time()
        w, b, support_vec = train_svm(X, y, c)

        # print("w:",w.shape,
        #       "b:", b.shape,
        #       "Number of Support Vectors:",
        #       np.unique(support_vec, return_counts = True)[1][1])
        # print("Training Time Taken: %s seconds" % (time.time() - start_time))
        cross_val_accuracy.append(test_svm(val_X, val_y, w, b[0]))
        time_taken.append(time.time() - start_time)

    c_index += 1

    avg_cross_val_accuracy.append(np.array(cross_val_accuracy).mean())

    print("Avg. Cross-Validation Accuracy:", np.array(cross_val_accuracy).mean())
    print("Avg. Time Taken:", np.array(time_taken).mean())


print('')
print("")
optimal_c_index = avg_cross_val_accuracy.index(max(avg_cross_val_accuracy))
print("Maximum value of cross-validation accuracy is obtained for C = ", C[optimal_c_index])
print("Maximum Cross-Validation Accuracy =", max(avg_cross_val_accuracy))

fold_ind = fold_indicies[optimal_c_index]
train_ind = train_indicies[optimal_c_index]
val_X = mean_centered_train_data[fold_ind]
val_y = train_label[fold_ind]
X = mean_centered_train_data[train_ind]
y = train_label[train_ind]
out_w, out_b, support_vec = train_svm(X, y, C[optimal_c_index])

print("Test Accuracy: ", test_svm(mean_centered_test_data, test_label, out_w, out_b[0]))
