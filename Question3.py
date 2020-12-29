#!/usr/bin/env python
# coding: utf-8

# # Read from the folders

# In[1]:


import os
import math
import numpy as np
import pickle
import re
import matplotlib.pyplot as plt


# ### Generate pre-defined dictionary from dic.dat

# In[2]:


pre_defined_words = []
with open('../hw3_data/dic.dat', 'r') as f:
    for word in f.readlines():
        pre_defined_words.append(word.strip())
f.close()


# In[3]:


def list_all_files(directory, extensions=None):
    for root, dirnames, filenames in os.walk(directory):
        for filename in filenames:
            base, ext = os.path.splitext(filename)
            joined = os.path.join(root, filename)
            if extensions is None or ext.lower() in extensions:
                yield joined


# In[4]:


def get_index(word):
    global pre_defined_words
    try:
        return pre_defined_words.index(word)
    except ValueError:
        return -1


# In[5]:


def examples_to_dataset(examples, block_size=2):
    X = []
    y = []
    for path, label in examples:
        # initialize a column vector
        bg_feature = np.zeros((len(pre_defined_words)))
        with open(path, 'r', encoding='ISO 8859-1') as f:
            raw_string = f.read()
            vector_of_strings = re.split("[\s.,?]", raw_string)
#             vector_of_strings = map(str.strip, vector_of_strings)
            
            for word in vector_of_strings:
                # get index of the word using pre-defined-dictionary
                ind = get_index(word)
                if ind != -1:
                    bg_feature[ind] += 1
        f.close()
        
        X.append(bg_feature)
        y.append(label)
        
    return np.array(X), np.array(y)


# ### Get Training Data

# In[6]:


negative_paths = list(list_all_files('../hw3_data/train/spam/', ['.txt']))
print('loaded', len(negative_paths), 'negative examples')
positive_paths = list(list_all_files('../hw3_data/train/ham/', ['.txt']))
print('loaded', len(positive_paths), 'positive examples')
#storing txt files as a tuple(path, class)
examples = [(path, 0) for path in negative_paths] + [(path, 1) for path in positive_paths]


# In[7]:


X_train, y_train = examples_to_dataset(examples)


# ### Get Testing Data

# In[8]:


negative_paths = list(list_all_files('../hw3_data/test/spam/', ['.txt']))
print('loaded', len(negative_paths), 'negative examples')
positive_paths = list(list_all_files('../hw3_data/test/ham/', ['.txt']))
print('loaded', len(positive_paths), 'positive examples')
#storing txt files as a tuple(path, class)
examples = [(path, 0) for path in negative_paths] + [(path, 1) for path in positive_paths]


# In[9]:


X_test, y_test = examples_to_dataset(examples)


# # Most Frequent Words - Training Data

# In[10]:


max_over_columns = X_train.sum(axis=0)
freq_words = {}
for word_index in max_over_columns.argsort()[-3:][::-1]:
    freq_words[pre_defined_words[word_index]] = max_over_columns[word_index]


# In[11]:


most_freq_words = [(k, v) for k, v in freq_words.items()]
print("Top three Frequent Words are as follows:")
print(most_freq_words)


# # Cross-Entropy Function & Gradient Descent

# In[12]:


def sigmoid(x):
    """
    Note that this function might throw a buffer-overflow Warning. 
    Numpy takes care of the approximation although the warning is thrown. 
    """
    return 1 / (1 + np.exp(-x))


# In[13]:


def cross_entropy(X, w, b, y, lambda_val = 0, with_regularization = True):
    """
    X.shape = N x d   ==> each element is of shape (d,)
    y.shape = (d,)
    w.shape = (d,)
    b => scalar
    lambda_val => scalar
    with_regularization is True by default. You can set it to False while passing the function argument. 
    """
    loss_value = 0
    for i in range(X.shape[0]):
        intermediate1 = sigmoid(np.dot(w, X[i]) + b)
        
        if intermediate1 < 1e-16:
            intermediate1 = 1e-16
        elif 1 - intermediate1 < 1e-16:
            intermediate1 = 1 - 1e-16
        
        loss_value += -( y[i] * np.log(intermediate1) + (1 - y[i]) * np.log(1 - intermediate1) )
    
    if with_regularization:
        loss_value += lambda_val*np.power(np.linalg.norm(w, ord=2), 2)
    
    return loss_value


# In[14]:


def gradient_descent(X, w, b, y, learning_rate = 0.01, lambda_val = 0, with_regularization = True):
    """
    X.shape = N x d   ==> each element is of shape (d,)
    y.shape = (d,)
    w.shape = (d,)
    b => scalar
    lambda_val => scalar
    with_regularization is True by default. You can set it to False while passing the function argument. 
    """
    w_temp = np.zeros(w.shape)
    b_temp = 0
    
    for i in range(X.shape[0]):
        intermediate1 = sigmoid(np.dot(w, X[i]) + b)
        
        if intermediate1 < 1e-16:
            intermediate1 = 1e-16
        elif 1 - intermediate1 < 1e-16:
            intermediate1 = 1 - 1e-16
        
        intermediate1 = intermediate1 - y[i]
        w_temp = np.add(w_temp, np.multiply(intermediate1, X[i]))
        
        b_temp += intermediate1
    
    if with_regularization:
        w_temp = np.add(w_temp, 2*lambda_val*w)
    
    w_new = w - (learning_rate * w_temp)
    b_new = b - (learning_rate * b_temp)
    
    return w_new, b_new


# In[15]:


# def cross_entropy(w, X, y, b, lambda_val = 0, with_regularization=True):
#     # w.shape = (d x 1)
#     # X.shape = (N x d)
#     # y.shape = (N x 1)
#     # b.shape = (N x 1)
#     # lambda = scalar value
#     X_w = np.dot(X, w)
#     intermediate_vec1 = X_w + b
#     tmp = sigmoid(intermediate_vec1)
    
#     # set all the zero-values to 1e-16 
#     tmp[np.where(tmp < 1e-16)[0]] = 1e-16
#     summation_first_term = np.dot(y.T, np.log(tmp))
    
#     tmp = 1 - sigmoid(intermediate_vec1)
#     tmp[np.where(tmp < 1e-16)[0]] = 1e-16
    
#     summation_second_term = np.dot(1-y.T, np.log(tmp))
    
#     summation = -1 * (summation_first_term + summation_second_term)
    
#     # add regularization term if regularization is involved in cost function
#     if with_regularization:
#         summation += lambda_val * np.linalg.norm(w)
    
#     return summation


# In[16]:


# def gradient_descent(X, y, w, b, learning_rate = 0.001, with_regularization = True, lambda_val = 0):
    
#     # update w_t+1
#     intermediate1 = sigmoid(np.dot(X, w) + b) - y
#     # matrix 
#     intermediate2 = intermediate1 * X
    
#     tmp = intermediate2.sum(axis=0).reshape(-1,1)
    
#     if with_regularization:
#         w = tmp + 2 * lambda_val * w
#     else:
#         w = tmp
        
#     # update b_t+1
#     b = intermediate1.sum() * np.ones((intermediate1.shape[0], 1))
#     return w, b


# In[17]:


total_iterations = 50
step_sizes = [0.001, 0.01, 0.05, 0.1, 0.5]
l2_norm = {}


# In[18]:


plt.figure(figsize=(16,8))
temp = []
for learning_rate in step_sizes:
    # initialize the values for weights and bias
    w = np.zeros((X_train.shape[1]))
    b = 0.1
    
    loss_values = []
    iteration_values = []
    for epoch in range(total_iterations):
        if learning_rate == 0.001:
            temp.append(b)
        loss_values.append(cross_entropy(X_train, w, b, y_train, with_regularization=False))
        iteration_values.append(epoch)
        w, b = gradient_descent(X_train, w, b, y_train, learning_rate = learning_rate, with_regularization=False)

    l2_norm[learning_rate] = np.linalg.norm(w, ord=2)
    
    plt.semilogy(iteration_values, loss_values, label = '\u03B7 = ' + str(learning_rate))

plt.legend(loc = 'best')
plt.title("Cross-Entropy vs Number of Steps - Without Regularization")
plt.xlabel('Number of Steps')
plt.ylabel("Loss Function Values")
plt.grid(True)
plt.show()

print(temp)

# In[19]:


l2_norm_step_size = [('\u03B7 = '+str(k), v) for k, v in l2_norm.items()]
print("Value of L2-norm of w for various values of step-sizes:")
print("{")
for x, y in l2_norm_step_size:
    print("(",x, ",", y, ")")
print("}")


# # 3.4 (c)

# In[20]:


total_iterations = 50
step_sizes = [0.001, 0.01, 0.05, 0.1, 0.5]
lambda_values = [0, 0.1, 0.2, 0.3, 0.4, 0.5]
l2_norm = {}


# In[21]:


plt.figure(figsize=(16,8))
for learning_rate in step_sizes:
    # initialize the values for weights and bias
    w = np.zeros((X_train.shape[1]))
    b = 0.1
    
    loss_values = []
    iteration_values = []

    for epoch in range(total_iterations):
        loss_values.append(cross_entropy(X_train, w, b, y_train, lambda_val=0.1))
        iteration_values.append(epoch)
        w, b = gradient_descent(X_train, w, b, y_train, learning_rate = learning_rate, lambda_val = 0.1)
        
    plt.semilogy(iteration_values, loss_values, label = '\u03B7 = ' + str(learning_rate))

plt.legend(loc = 'best')
plt.title("Cross-Entropy vs Number of Steps ( \u03BB = 0.1 )")
plt.xlabel('Number of Steps')
plt.ylabel("Loss Function Values")
plt.show()


# In[22]:


l2_norm = {}
for lambda_val in lambda_values:
    learning_rate = 0.01
    loss_values = []
    iteration_values = []
    
    # initialize the values for weights and bias
    w = np.zeros((X_train.shape[1]))
    b = 0.1

    for epoch in range(total_iterations):
        loss_values.append(cross_entropy(X_train, w, b, y_train, lambda_val=lambda_val))
        iteration_values.append(epoch)
        w, b = gradient_descent(X_train, w, b, y_train, learning_rate = 0.01, lambda_val = lambda_val)
            
    l2_norm[lambda_val] = np.linalg.norm(w, ord=2)


# In[23]:


l2_norm_lambda = [('\u03BB = '+str(k), v) for k, v in l2_norm.items()]
print("Value of L2-norm of w for various values of lambda:")
print("{")
for x, y in l2_norm_lambda:
    print("(",x, ",", y, ")")
print("}")

