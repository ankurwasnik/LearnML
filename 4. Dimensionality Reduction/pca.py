import numpy as np
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score as accuracy
#all required libraries are imported

'''Author: Ankur W [LearnML} '''
def loadData():	
	df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data' , header=None)
	df.columns =['Class_label','Alcohol','Malic acid','Ash','Alkalinity of ash','Magnesium' , 'Total phenols' ,'Flavanoids','Nonflavanoids phenols','Proanthocyanins','Color Intensity','Hue','OD280/OD315 of diluted wines','Proline']
	X,y = df.iloc[:,1:].values , df.iloc[:,0].values
	Xtrain ,Xtest , ytrain,ytest = train_test_split(X,y ,test_size=0.3 ,random_state=0,stratify=y)
	return Xtrain,Xtest,ytrain,ytest


if __name__=='__main__' :
	X_train , X_test , Y_train, Y_test = loadData()
	#initializing the pca analyzer
	pca = PCA(n_components=2)
	#logistic regressor estimator
	log_reg = LogisticRegression(multi_class='ovr' , random_state=1 ) 
	#dimensionality reduction
	X_train_pca = pca.fit_transform(X_train)
	X_test_pca  = pca.transform(X_test)
	#fitting the logistic regressor on reduced data in terms of dimensionality
	log_reg.fit(X_train_pca,Y_train)
	#predicting on reduced test data
	predictions = log_reg.predict(X_test_pca)
	#evaluating reduced test data
	print('Evaluation of PCA test dataset: ', accuracy(Y_test,predictions))
	
