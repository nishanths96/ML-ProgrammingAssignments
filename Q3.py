#!/usr/bin/env python
# coding: utf-8

# # Read from the folders

# In[1]:


import os
import math
import numpy as np
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
    X = np.zeros((len(examples), len(pre_defined_words)))
    y = np.empty((len(examples), 1))
    
    i = 0
    for path, label in examples:
        with open(path, 'r', encoding='ISO 8859-1') as f:
            raw_string = f.read()
            vector_of_strings = re.split("[\s.,?]", raw_string)
#             vector_of_strings = map(str.strip, vector_of_strings)
            
            for word in vector_of_strings:
                # get index of the word using pre-defined-dictionary
                ind = get_index(word)
                if ind != -1:
                    X[i][ind] += 1
        f.close()
        y[i] = label
        i += 1
        
    return X, y


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


def sigmoid(inp):
    value = 1 / ( 1 + np.exp(-inp) )
    
    for i in range(value.shape[0]):
        if value[i] < 1e-16: 
            value[i] = 1e-16
        elif 1 - value[i] < 1e-16: 
            value[i] = 1 - 1e-16
    return value


# In[13]:


def cross_entropy(w, b, X, y, lambda_val = 0, with_regularization = False):
    sigma_val = sigmoid(b + np.dot(X, w))

    summation = -( y * np.log(sigma_val) + (1-y) * np.log(1 - sigma_val))
    summation = np.sum(summation)
    # summation = -(np.dot(y.T, np.log(sigma_val)) + np.dot((1-y.T), np.log(1-sigma_val)))
    
    if with_regularization:
        summation = summation + (lambda_val * np.power(np.linalg.norm(w, ord=2), 2))
 
    intermediate_vec = (y - sigma_val)    
    dw = np.dot(intermediate_vec.T, X)
    
    if with_regularization:
        dw = dw - (2 * lambda_val * w.T)
        
    return summation, intermediate_vec, dw


# In[14]:


def gradient_descent(X, y):
    step_sizes = [0.001, 0.01, 0.05, 0.1, 0.5]
    
    loss_values = []
    l2_norm = {}
    temp = []
    for lr in step_sizes:
        weights = np.zeros((X.shape[1], 1))
        bias = 0.1
        for step in range(50):
            if lr == 0.001:
                temp.append(bias)

            summation, db, dw = cross_entropy(weights, bias, X, y)
            weights = weights + lr * dw.T
            bias = bias + lr*db.sum()

            loss_values.append(summation)
        l2_norm[lr] = np.linalg.norm(weights, ord=2)
    
    return l2_norm, temp


l2_norm, bias_val = gradient_descent(X_train, y_train)

print(bias_val)
# In[ ]:




