# -*- coding: utf-8 -*-
"""
Created on Thu May 21 21:14:39 2020

@author: kingslayer
"""

#importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#importing the dataset
dataset=pd.read_csv("Creditcard.csv")

X=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

#Splitting
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)

#feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X=StandardScaler()
X_train=sc_X.fit_transform(X_train)
X_test=sc_X.transform(X_test)
X=sc_X.transform(X)

#ANN

from keras.models import Sequential
from keras.layers import Dense,Dropout

classifier=Sequential()

classifier.add(Dense(output_dim=15,init="uniform",activation="relu",input_dim=30))

classifier.add(Dense(output_dim=15,init="uniform",activation="relu"))
classifier.add(Dropout(0.2))

classifier.add(Dense(output_dim=15,init="uniform",activation="relu"))
classifier.add(Dropout(0.2))

classifier.add(Dense(output_dim=15,init="uniform",activation="relu"))
classifier.add(Dropout(0.2))

classifier.add(Dense(output_dim=1,init="uniform",activation="sigmoid"))

classifier.compile(optimizer="adam",loss="binary_crossentropy",metrics=["accuracy"])

classifier.fit(X_train,y_train,batch_size=100,nb_epoch=10)

y_pred=classifier.predict(X_test)

y_pred=(y_pred>0.5)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

from sklearn.metrics import confusion_matrix
dataset2=dataset.drop(columns="Class")
datase2=sc_X.transform(dataset2)
cm1=confusion_matrix(dataset["Class"],classifier.predict(dataset2).round())

#Random Forest-->Winner



from sklearn.ensemble import RandomForestClassifier as RFC
classifier_rf=RFC(n_estimators=100)
classifier_rf.fit(X_train,y_train)

y_pred=classifier_rf.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

from sklearn.metrics import confusion_matrix
dataset2=dataset.drop(columns="Class")
datase2=sc_X.transform(dataset2)
cm1=confusion_matrix(dataset["Class"],classifier_rf.predict(dataset2))



#Decision Tree
from sklearn.tree import DecisionTreeClassifier as DTC
classifier=DTC()
classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

from sklearn.metrics import confusion_matrix
dataset2=dataset.drop(columns="Class")
datase2=sc_X.transform(dataset2)
cm1=confusion_matrix(dataset["Class"],classifier.predict(dataset2))




#Balancing




#UnderSample
import random
fraud_indices=dataset[dataset.Class==1].index
normal_indices=dataset[dataset.Class==0].index

normal_indices=np.random.choice(normal_indices,len(fraud_indices))
fraud_indices=np.asarray(fraud_indices)

new_data=np.concatenate((normal_indices,fraud_indices))

data=dataset.iloc[new_data,:]

X_new=data.iloc[:,data.columns!="Class"]
y_new=data.iloc[:,data.columns=="Class"]

#Splitting
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X_new,y_new,test_size=0.2,random_state=0)



from sklearn.ensemble import RandomForestClassifier as RFC
classifier=RFC(n_estimators=100)
classifier.fit(X_train,y_train)

y_pred=classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

from sklearn.metrics import confusion_matrix
dataset2=dataset.drop(columns="Class")
datase2=sc_X.transform(dataset2)
cm1=confusion_matrix(dataset["Class"],classifier.predict(dataset2))



#OverSample

from imblearn.over_sampling import SMOTE
x_resample,y_resample=SMOTE().fit_sample(X,y.values.ravel())

x_resample=pd.DataFrame(x_resample)
y_resample=pd.DataFrame(y_resample)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(x_resample,y_resample,test_size=0.2,random_state=0)

classifier_rf.fit(X_train,y_train)

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)


y_pred=classifier_rf.predict(X_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

from sklearn.metrics import confusion_matrix
y_expected=pd.DataFrame(y)
cm1=confusion_matrix(y_expected,classifier_rf.predict(X))

