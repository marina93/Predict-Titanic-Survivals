#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 11 09:27:43 2020

@author: marina
"""
import pandas as pd
import sklearn.metrics as metrics
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,classification_report

from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from collections import Counter

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
#train.info()

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
       
    logreg = LogisticRegression()
    logreg.fit(X_train, y_train)
    y_pred1 = logreg.predict(X_train)
    y_pred2 = logreg.predict(X_val)
    
    probs = logreg.predict_proba(X_val)
    preds = probs[:,1]
    fpr, tpr, thresholds = metrics.roc_curve(y_val, preds)
    roc_auc = metrics.auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    print("Logistic Regression Classifier distribution: {}".format(Counter(y)))
    print(classification_report(y_val, y_pred2))
    print('\n')
    print('Logistic Regression Classifier')
    print('Train accuracy score: ', accuracy_score(y_train, y_pred1))
    print('Test accuracy score: ', accuracy_score(y_val, y_pred2))
    print('\n')
    
    
def random_forest():
    
    rfc = RFC()
    rfc.fit(X_train, y_train)
    y_pred1 = rfc.predict(X_train)
    y_pred2 = rfc.predict(X_val)
    
    probs = rfc.predict_proba(X_val)
    preds = probs[:,1]
    fpr, tpr, thresholds = metrics.roc_curve(y_val, preds)
    roc_auc = metrics.auc(fpr, tpr)
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
    print("Random Forest Classifier distribution: {}".format(Counter(y)))
    print(classification_report(y_val, y_pred2))
    print('\n')
    print('Random Forest Classifier')
    print('Train accuracy score: ', accuracy_score(y_train, y_pred1))
    print('Test accuracy score: ', accuracy_score(y_val, y_pred2))
    print('\n')
    
def ridge_regression():
    rrc = Ridge()
    rrc.fit(X_train, y_train)
    y_prob = rrc.predict(X_train)
    # Ridge outputs probs, make conversion to get the actual prediction
    y_pred1 = np.asarray([np.argmax(prob) for prob in y_prob])
    y_prob = rrc.predict(X_val)
    y_pred2 = np.asarray([np.argmax(prob) for prob in y_prob])
        
    print('Ridge Regression Classifier')
    print('Train accuracy score: ', accuracy_score(y_train, y_pred1))
    print('Test accuracy score: ', accuracy_score(y_val, y_pred2))
    print('\n')

def kNeighbors():
    knn = KNeighborsClassifier()
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_train)
    y_pred2 = knn.predict(X_val)
    print('K Nearest Neighbors Classifier')
    print('Train accuracy score: ', accuracy_score(y_train, y_pred))
    print('Test accuracy score: ', accuracy_score(y_val, y_pred2))
    print('\n')
   
def xgboost():    
    # DMatrix: Internal data structure used to optimize training speed and memory efficiency
    xg_train = xgb.DMatrix(X_train, label=y_train)
    xg_val = xgb.DMatrix(X_val, label=y_val)
    xg_train.save_binary('train.buffer')
    xg_val.save_binary('test.buffer')
    
    #setup parameters for xgboost
    params = {}
    # classification algorithm
    params['num_class'] = 2
    #List of validation sets for which metrics will evaluated during training. Validation metrics will help us track the performance of the model.
    #watchlist = [(xg_train, 'train'), (xg_val, 'val')]
    bst = xgb.train(params, xg_train, 30)
    y_pred1 = bst.predict(xg_train)
    y_pred2 = bst.predict(xg_val)
    print('XGBoost Classifier')
    print('Train accuracy score: ', accuracy_score(y_train, y_pred1))
    print('Test accuracy score: ', accuracy_score(y_val, y_pred2))
    print('\n')
    

def main():
    logistic_regression()
    random_forest()
    ridge_regression()
    kNeighbors()
    xgboost()

if __name__ == "__main__":
    main()




     
     



     
     


