# -*- coding: utf-8 -*-
"""
Created on Thu Jul 24 11:39:23 2025

@author: Admin
"""

### to build a machinelearning model that can predict wheather a person is# 
#### likely to have heart disease or not base on various mesical parameters#

#import nessory libary files##
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix,classification_report
### load the dataset
data = pd.read_csv("C:/Users/Admin/Downloads/archive (12)/heart.csv")

## basic information
data.dtypes 
data.astype
print ("shape of data:",data.shape)
print (data.head())

# corealtion heatmap
plt.figure(figsize=(10,7))
sns.heatmap(data.corr(), annot=True,cmap='coolwarm')
plt.title("feature correlation")
plt.show()
# features and target
X = data.drop('target',axis=1)# all input features
y = data['target'] #output label

#train_test_split(split the data 80% traning,20%testing)
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
#scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
#model
model = RandomForestClassifier()
model.fit(X_train,y_train)
y_pred= model.predict(X_test)
print('accuracy:',accuracy_score(y_test,y_pred))
print('confusion matrix\n',confusion_matrix(y_test,y_pred))
print('classification report\n',classification_report(y_test,y_pred))

### i build a machine learning model using python to predict heart
## disease i used preprocessing,visualization,random forestalgorithem
## and evaluated the model with accuracy,confusion matrix,and 
##classificaion report.this project taught me how to handle real-world
##medical data and apply ml in healthcare problem











