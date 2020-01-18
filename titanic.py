#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 09:27:43 2020

@author: marina
"""
import pandas as pd
import sklearn.metrics as metrics
import numpy as np


from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.linear_model import Ridge

#1. Extract data into labels and features
train = pd.read_csv('./train.csv')
test = pd.read_csv('./test.csv')


# Access fields:  train.columns
# Access first column: train.iloc[:,0]

# Check number of classes 
train['Survived'].value_counts(normalize=True)

# Ignore features with high cardinality
max_cardinality = 100
high_cardinality = [col for col in train.select_dtypes(exclude = np.number) if train[col].nunique()>max_cardinality]
train = train.drop(columns=high_cardinality)
train = train.dropna()
train = train.sort_values("Age", ascending=True)
train.info()

X = train.loc[:, train.columns!='Survived']
y = train['Survived']

# Next: Change OBJECTS to INTS: OneHotEncoder or Pandas get_dummies()
X = pd.get_dummies(X)

X.drop(columns='PassengerId')



# Scale features to ensure that they are zero-centered. 
# Ensure variances of the features are in the same range.
X = scale(X)
#X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.75, test_size=0.25, random_state=42, shuffle=True)
X_train, X_val, y_train, y_val = train_test_split(
        X, y, train_size=0.75, test_size=0.25, random_state=42, shuffle=True)

# 3. Classifiers
# 3.1 Logistic Regression
def logistic_regression():
    
# =============================================================================
#      PCA (Principal Component Analysis) Reduce dimensionality 
#      by creating new variables as linear combinations of the original ones. 
#      Only a few of the new variables are used, and the old ones I think they are 
#      not used any more. The new variables contain more information than the old 
#      simpel ones
# =============================================================================
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    print('Logistic Regression Classifier')
    print('Train accuracy score: ',accuracy_score(y_train,logreg.predict(X_train)))
    print('Test accuracy score: ',accuracy_score(y_val, logreg.predict(X_val)))
    print('\n')
    
def random_forest_classifier():
    
    rfc = RFC()
    rfc.fit(X_train, y_train)
    print('Random Forest Classifier')
    print('Train accuracy score: ', accuracy_score(y_train, rfc.predict(X_train)))
    print('Test accuracy score: ', accuracy_score(y_val, rfc.predict(X_val)))
    print('\n')
    
def ridge_regression():
    rrc = Ridge()
    rrc.fit(X_train, y_train)
    y_prob = rrc.predict(X_train)
    # Ridge outputs probs, make conversion to get the actual prediction
    y_pred = np.asarray([np.argmax(prob) for prob in y_prob])
    y_prob = rrc.predict(X_val)
    y_pred2 = np.asarray([np.argmax(prob) for prob in y_prob])
    
    print('Ridge Regression Classifier')
    print('Train accuracy score: ', accuracy_score(y_train, y_pred))
    print('Test accuracy score: ', accuracy_score(y_val, y_pred2))
    print('\n')
    
    
def main():
    logistic_regression()
    random_forest_classifier()
    ridge_regression()
     

if __name__ == "__main__":
    main()




     
     



     
     


