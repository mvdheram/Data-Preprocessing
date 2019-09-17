#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 18:09:24 2018

@author: Meher
"""
# Data Preprocessing Template

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
#iloc.[rows,columns] 
dataset = pd.read_csv('Data.csv')
x = dataset.iloc[:,:-1].values # Matrix of features/independent variables
y = dataset.iloc[:, 3].values # vector of dependent; A vector is a list of numbers (can be in a row or column)

# values need to be replaced by NaN for the transformation to take place
#stratefy can be mean(average),median(middle value),most_frequent(mode),axis=0,(impute columns) else 1 (impute rows) 

# taking care of missing data
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN',strategy = 'mean', axis = 0) # creating object of imputation with parameters
imputer = imputer.fit(x[:, 1:3]) # passing the data to imputation function 
x[:, 1:3]= imputer.transform(x[:, 1:3])  # Assigning the imputation to the data

# Since machine learning models are based on mathematical equations we need to transform\encode "Categorical" data \text to numbers

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_x = LabelEncoder() # Creating an object of LabelEncoder class
x[:, 0] = labelencoder_x.fit_transform(x[:, 0]) # assign to the data the applied transfomation 
 
# problem is that ML models would think that there is a relation between the encodings [2>1>0] but that should noot be the case
# Solution : Dummy Encoding {num of colums = number of categories}

onehotencoder = OneHotEncoder(categorical_features = [0]) #transforming column[0]
x = onehotencoder.fit_transform(x).toarray()
#dependent variables 
labelencoder_y = LabelEncoder() #transforming column[3]
y = labelencoder_y.fit_transform(y)



# Splitting the dataset into the training set and Test set
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2, random_state = 0 ) #  test_size percentage of data for test 

# Feature Scaling used to normalise the data so that one feature value is not dominated by other 
from sklearn.preprocessing import StandardScaler
sc_x = StandardScaler()
x_train = sc_x.fit_transform(x_train)
x_test = sc_x.transform(x_test)

# Why do we need to fit and transform for training and transform for the test?
#feature scaling two types:
#Standardisation and Normalisation 
# To center the data (make it have zero mean and unit standard error), you subtract the mean and then divide the result by the standard deviation.
#x′(standardisation)=x(data)−μ(mean)/σ(standard deviation)
#mean= sum /total number of events
#standard Deviation = Squre root of average of sqared differences from the mean (x - mean)square/n
# Every sklearn's transform's fit() just calculates the parameters (e.g. μ and σ in case of StandardScaler) and saves them as an internal objects state. Afterwards, you can call its transform() method to apply the transformation to a particular set of examples
# why scaling ?? Refer IR(Eucledian distance)  

#Normalisation x'= x-min of all the feature values(x)/max of feature values(x)- min of feature values(x)


 # why no Feature scaling to y_train and y_test??
 # The present dataset is a classification problem with categorical dependent varible y (Categorical variables take on values that are names or labels) 
 # in Regression we apply scaling to dependent variables as values are Qunatitative (numbers,qunatities) 