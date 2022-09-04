




#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 14:15:45 2021

@author: bf101616
"""





import numpy as np

from sklearn.model_selection import StratifiedKFold
import numpy
# fix random seed for reproducibility




import math


# import xgboost as xgb
# from xgboost.sklearn import XGBClassifier
# from xgboost import XGBClassifier


from sklearn.impute import SimpleImputer
from pprint import pprint


#import pandas as pd
from sklearn.preprocessing import MinMaxScaler
#from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
#from sklearn.cross_validation import train_test_split 
import numpy as np 

#train_data = pd.read_csv("train.csv")
#test_data = pd.read_csv("test.csv")

#y_train = train_data["Survived"]
#train_data.drop(labels="Survived", axis=1, inplace=True)

#full_data = train_data.append(test_data)
#drop_columns = ["Name", "Age", "SibSp", "Ticket", "Cabin", "Parch", "Embarked"]
#full_data.drop(labels=drop_columns, axis=1, inplace=True)
#full_data = pd.get_dummies(full_data, columns=["Sex"])
#full_data.fillna(value=0.0, inplace=True)
#X_train = full_data.values[0:891]
#X_test = full_data.values[891:]


#X=data.loc[0:880,'Group':'Vmax'].values 802
#X=X.replace('',np.nan)
#import matplotlib.pyplot as plt
#import seaborn as sns
# roc curve and auc score
from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
#import Tkinter as tk


from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer



from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report
#from sklearn.grid_search import GridSearchCV

import numpy as np 
#import matplotlib.pyplot as plt 



from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection  import train_test_split  #mohemen in shekli bashe






import pandas as pd
#data=pd.read_csv("prebp40.csv",delimiter=";" ,decimal="."),sep='\t'
data=pd.read_csv("Data2.csv"  )


#from sklearn.model_selection import train_test_split

#
#S	1
#Es	0
#S	1
#I	0
#A	1
#N	0


# data['Group']=data['Group'].replace('Nice',0)
# data['Group']=data['Group'].replace('Helsinki',1)
# data['Group']=data['Group'].replace('FINLAND',2)

 
data['sexo']=data['sexo'].replace('H',0)
data['sexo']=data['sexo'].replace('V',1)


data['canal_entrada']=data['canal_entrada'].replace('KHL',0)


data['canal_entrada']=data['canal_entrada'].replace('KHE',1)
data['canal_entrada']=data['canal_entrada'].replace('KHD',2)
data['canal_entrada']=data['canal_entrada'].replace('KFC',3)
data['canal_entrada']=data['canal_entrada'].replace('KAT',4)


data['indfall']=data['indfall'].replace('N',0)

data['nomprov']=data['nomprov'].replace('MALAGA',0)
data['nomprov']=data['nomprov'].replace('MALAGA',0)



#data['Weekly_othersporttrainingtime']=data['Weekly_othersporttrainingtime'].replace('Di11icile',	2)
#data['Weekly_othersportrainingintensity']=data['Weekly_othersportrainingintensity'].replace('Di11icile',	2)
#data['Weekly_competitionintensity']=data['Weekly_competitionintensity'].replace('Di11icile',	2)
#	Weekly_othersporttrainingtime	Weekly_othersportrainingintensity



print('coloumn and rows:', data.shape)
print(data.size)


X=data.loc[2:31]
Y=data.loc[1]








####





# Splitting the dataset into train and test 
X_train, X_test, y_train, y_test = train_test_split(X,Y,test_size = 0.3, random_state = 100) 
      



scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

state = 12  
test_size = 0.30  
  
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,  
    test_size=test_size, random_state=state)


#Import libraries
#import pandas as pd
import numpy as np
from bayes_opt import BayesianOptimization
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score


# Function to split the dataset 
def splitdataset(X,Y): 
  
    # Separating the target variable 
    #X = balance_data.values[:, 1:5] 
    #X=X75
    #Y = balance_data.values[:, 0]
    #Y=Y75 
  
    # Splitting the dataset into train and test 
    X_train, X_test, y_train, y_test = train_test_split( 
    X,Y,test_size = 0.3, random_state = 100) 
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    state = 12  
    test_size = 0.30  
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train,  
    test_size=test_size, random_state=state)

      
    return  X_train, X_test, y_train, y_test 
      
#Bayesian optimization
def bayesian_optimization(X_train, y_train, X_val, y_val, function, parameters):
   #X_train, y_train, X_test, y_test = dataset
  # X_train, y_train, X_test, y_test = dataset
   n_iterations = 5
   gp_params = {"alpha": 1e-4}

   BO = BayesianOptimization(function, parameters)
   BO.maximize(n_iter=n_iterations, **gp_params)

   return BO.max

def rfc_optimization(cv_splits):
    def function(n_estimators, max_depth, min_samples_split):
        return cross_val_score(
               RandomForestClassifier(
                   n_estimators=int(max(n_estimators,0)),                                                               
                   max_depth=int(max(max_depth,1)),
                   min_samples_split=int(max(min_samples_split,2)), 
                   n_jobs=-1, 
                   random_state=42,   
                   class_weight="balanced"),  
               X=X_train, 
               y=y_train, 
               cv=cv_splits,
               scoring="roc_auc",
               n_jobs=-1).mean()

    parameters = {"n_estimators": (10, 1000),
                  "max_depth": (1, 150),
                  "min_samples_split": (2, 10)}
    
    return function, parameters

#Train model
def train(X_train, y_train, X_test, y_test, function, parameters):
    dataset = (X_train, y_train, X_val, y_val)
    cv_splits = 4
    
    best_solution = bayesian_optimization(X_train, y_train, X_val, y_val, function, parameters)      
    params = best_solution["params"]

    model = RandomForestClassifier(
             n_estimators=int(max(params["n_estimators"], 0)),
             max_depth=int(max(params["max_depth"], 1)),
             min_samples_split=int(max(params["min_samples_split"], 2)), 
             n_jobs=-1, 
             random_state=42,   
             class_weight="balanced")

    model.fit(X_train, y_train)
    
    return model

def xgb_optimization(cv_splits, eval_set):
    def function(eta, gamma, max_depth):
            return cross_val_score(
                   xgb.XGBClassifier(
                       objective="binary:logistic",
                       learning_rate=max(eta, 0),
                       gamma=max(gamma, 0),
                       max_depth=int(max_depth),                                               
                       seed=42,
                       nthread=-1,
                       scale_pos_weight = len(y_train[y_train == 0])/
                                          len(y_train[y_train == 1])),  
                   X=X_train, 
                   y=y_train, 
                   cv=cv_splits,
                   scoring="roc_auc",
                   fit_params={
                        "early_stopping_rounds": 10, 
                        "eval_metric": "auc", 
                        "eval_set": eval_set},
                   n_jobs=-1).mean()

    parameters = {"eta": (0.001, 0.4),
                  "gamma": (0, 20),
                  "max_depth": (1, 2000)}
    
    return function, parameters



def xgb_optimization(cv_splits, eval_set):
    def function(eta, gamma, max_depth):
            return cross_val_score(
                   xgb.XGBClassifier(
                       objective="binary:logistic",
                       learning_rate=max(eta, 0),
                       gamma=max(gamma, 0),
                       max_depth=int(max_depth),                                               
                       seed=42,
                       nthread=-1,
                       scale_pos_weight = len(y_train[y_train == 0])/
                                          len(y_train[y_train == 1])),  
                   X=X_train, 
                   y=y_train, 
                   cv=cv_splits,
                   scoring="roc_auc",
                   fit_params={
                        "early_stopping_rounds": 10, 
                        "eval_metric": "auc", 
                        "eval_set": eval_set},
                   n_jobs=-1).mean()

    parameters = {"eta": (0.001, 0.4),
                  "gamma": (0, 20),
                  "max_depth": (1, 2000)}
    
    return function, parameters




def main(): 
      
    # Building Phase 
    X_train, X_test, y_train, y_test=splitdataset(X,Y)
    cv_splits = 4
    eval_set=[(X_train, y_train), (X_val, y_val)]

    function, parameters=rfc_optimization(cv_splits)
    function2, parameters2=xgb_optimization(cv_splits,eval_set)
    #Bo=bayesian_optimization(X_train, y_train, X_val, y_val, function, parameters)
    model=train(X_train, y_train, X_test, y_test, function, parameters)
    #model2=train(X_train, y_train, X_test, y_test, function2, parameters2)

    

    #pprint(model.best_estimator_.get_params())
#print(model.best_score)
    lr_list = [0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]

    for learning_rate in lr_list:



        print("Learning rate: ", learning_rate)
        print("Accuracy score (training): {0:.3f}".format(model.score(X_train, y_train)))
        print("Accuracy score (validation): {0:.3f}".format(model.score(X_val, y_val)))





    predictions = model.predict(X_val)

    print("Confusion Matrix:")
    print(confusion_matrix(y_val, predictions))

    print("Classification Report")
    print(classification_report(y_val, predictions))

  #   probs = model.predict_proba(X_val)
  #   probs = probs[:, 1]
  #   fpr, tpr, thresholds = roc_curve(y_val, probs)
  #  # plot_roc_curve(fpr, tpr)




    xgb_clf = XGBClassifier()
    xgb_clf.fit(X_train, y_train)
    score = xgb_clf.score(X_val, y_val)
    print(score)

    for learning_rate in lr_list:



        print("Learning rate: ", learning_rate)
        print("Accuracy score (training): {0:.3f}".format(xgb_clf.score(X_train, y_train)))
        print("Accuracy score (validation): {0:.3f}".format(xgb_clf.score(X_val, y_val)))





    predictions2 = xgb_clf.predict(X_val)

    print("Confusion Matrix:")
    print(confusion_matrix(y_val, predictions2))

    print("Classification Report")
    print(classification_report(y_val, predictions2))



    probs = xgb_clf.predict_proba(X_val)
    probs = probs[:, 1]
    #fpr, tpr, thresholds = roc_curve(y_val, probs)
   # plot_roc_curve(fpr, tpr)
   



































########################################################   Second approach       ##########################################################

from sklearn.model_selection import GridSearchCV
#from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
model_params = {
    'n_estimators': [50, 150, 250],
    'max_features': ['sqrt', 0.25, 0.5, 0.75, 1.0],
    'min_samples_split': [2, 4, 6]
}

lr_list = [0.05, 0.075, 0.1, 0.25, 0.5, 0.75, 1]

# create random forest classifier model
rf_model = RandomForestClassifier(random_state=1)

# set up grid search meta-estimator
clf3 = GridSearchCV(rf_model, model_params, cv=5)

# train the grid search meta-estimator to find the best model
model = clf3.fit(X_train, y_train)

# print winning set of hyperparameters
pprint(model.best_estimator_.get_params())
#print(model.best_score)

for learning_rate in lr_list:



    print("Learning rate: ", learning_rate)
    print("Accuracy score (training): {0:.3f}".format(clf3.score(X_train, y_train)))
    print("Accuracy score (validation): {0:.3f}".format(clf3.score(X_val, y_val)))





predictions = clf3.predict(X_val)

print("Confusion Matrix:")
print(confusion_matrix(y_val, predictions))

print("Classification Report")
print(classification_report(y_val, predictions))

      
      
# Calling main function 
if __name__=="__main__": 
    main() 
















