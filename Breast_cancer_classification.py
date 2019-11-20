# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 15:12:14 2019

@author: Arumugam
"""

# Lib import

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.datasets import load_breast_cancer


# Data set load from sklearn inbuild dataset
data=load_breast_cancer()

# the breast cancer dataset is containt more keys of data
# that is DESCR ,data,feature_names ,filename,target,target_names

# so need to create csv datafile is data_features with target .

data_features=data['data']
data_feature_name=data['feature_names']
lable=data['target']
lable_names=data['target_names']

# we want to write our dataset to csv format in folder

df_data=pd.DataFrame(data_features,columns=data_feature_name)
df_data['cancer_class']=lable
df_data.to_csv('Breast_cancer_dataset.csv')

# Analyse the dataset

df_data.head(5)

df_data.describe()  # some feature is having low mean and variance value

df_data.isnull().sum()  # here don't have any null values

df_data.dtypes   # here don't have any cetegorical data onlynemerical values only is there

sns.pairplot(df_data,hue='cancer_class',vars=['mean radius','mean texture','mean smoothness','perimeter error']) # here see the each feature data representation 

sns.heatmap(df_data.corr(),annot=True,cmap='RdYlGn')

# Find the data features and lables

X=df_data.drop(['cancer_class'],axis=1)
Y=df_data['cancer_class']

# Apply standard scaler function for standardization of values
scaler=StandardScaler()
Final_data=scaler.fit_transform(X)

# split the data into train and testing part

X_train, X_test, y_train, y_test = train_test_split(X,Y, test_size=0.30, random_state=0)

# Build SVM Model
SVM_clf=SVC()

# Train the model using training data
Model=SVM_clf.fit(X_train,y_train) # Training process completed 


# Prediction process


Pred=Model.predict(X_test)
acc=Model.score(X_test,y_test)
print('Accuary =',acc)

# Validation process

from sklearn.metrics import confusion_matrix,classification_report


confusion_mat=confusion_matrix(y_test,Pred)
print('confusion_mat =',confusion_mat)


clf_report=classification_report(y_test,Pred)
print('confusion_mat =',clf_report)

# Cross validation using Gridsearch function

param_grid={'C':[0.1,1,10,100,100],'gamma':[1,0.1,0.01,0.001,0.0001],'kernel':['rbf']}

from sklearn.model_selection import GridSearchCV

graid_cv=GridSearchCV(SVC(),param_grid,refit=True,verbose=3)

graid_cv.fit(X_train,y_train)

graid_cv.best_params_  # so here we can find best parameters

graid_cv.best_estimator_  # here we can get best hyperparameters

graid_cv.best_score_     #  here we can get best score

pred_2=graid_cv.predict(X_test)

acc2=graid_cv.score(X_test,y_test)
print('Accuary =',acc2)


print('confusionmatric =',confusion_matrix(y_test,pred_2))
print('confusionmatric =',classification_report(y_test,pred_2))

# Finally we can got good accuracy using graid_seachcv method

# now we can test it.

#
# Find the Finnal parameters for SVM
SVM_clf_2=SVC(C=10, cache_size=200, class_weight=None, coef0=0.0,\
    decision_function_shape='ovr', degree=3, gamma=0.0001, kernel='rbf',\
    max_iter=-1, probability=False, random_state=None, shrinking=True,\
    tol=0.001, verbose=False)  # here we call best params

Model_2=SVM_clf_2.fit(X_train,y_train) # Training process completed 


# Prediction process

# overfitting test
from sklearn.metrics import accuracy_score
train_pred=Model_2.predict(X_train)

accuracy_score(y_train, train_pred)

# 
Pred_22=Model_2.predict(X_test)
acc_2=Model_2.score(X_test,y_test)
print('Accuary =',acc_2)   # both are same 

#sns.distplot((y_test-Pred_22),bins=50)    # Error value finding with displot


# Thanks for Reading 





