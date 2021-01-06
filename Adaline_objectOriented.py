'''
Author: Ankur Wasnik
Date 6th Jan 2020 22:17pm IST
Adaline means Adaptive linear neurons model
Difference between Perceptron and Adaline "
	1. Perceptron compares true labels with net input. While , Adaline compares the output of activation function .
	2. In Perceptron, weights are updated based on each training data. While, in adaline , weights are updated after each training sessions

'''
import numpy as np

class MyAdalineModel(object):
	'''

	'''
	def __init__(self, learning_rate=0.01 , epochs=10 , random_state=1):
		self.learning_rate=learning_rate
		self.epochs = epochs
		self.random_state=random_state
	
	def fit(self,X,y):
		rgen=np.random.RandomState(self.random_state)
		self.w_ = rgen.normal(loc=0.0,scale=0.01,size=1+X.shape[1])
		self.errors_ =[]
		
		for _ in range(self.epochs):
			net_input = self.net_input(X)
			y_hat = self.activation(net_input)
			errors = (y_hat - y)
			self.w_[0]+=self.learning_rate*(errors.sum())
			self.w_[1:]+= self.learning_rate*(np.dot(X.T,errors))
			cost = -y*(np.log(y_hat)) - (1-y)*(np.log(1-y_hat))
			self.errors_.append(cost)
		return self
	
	def net_input(self,X):
		return np.dot(X,self.w_[1:])+self.w_[0]
	
	def activation(self,X):
		return 1.0/(1.0+np.exp(-X))
	def predict(self,X,threshold=0.2):
		return np.where( self.activation(self.net_input(X)) >= threshold , 1 , -1)

if __name__=='__main__':
	import os 
	import pandas as pd
	s=os.path.join('https://archive.ics.uci.edu','ml','machine-learning-databases','iris','iris.data')
	print('Downloading Iris Dataset from :\n',s)
	df = pd.read_csv(s,header=None,encoding='utf-8')
	y=df.iloc[0:100,4].values
	y=np.where(y=='Iris-setosa',-1,1)
	X=df.iloc[0:100,[0,2]].values
	AdalineModel = MyAdalineModel(learning_rate=0.001 , epochs=50 , random_state=69)
	AdalineModel.fit(X,y)
	#We are done with training and You can predict on our trained model.
	''' Testing our trained data with one of trained data sample \nBut you are free to do on testing data.'''
	y_test = df.iloc[132 , 4]
	y_test = np.where(y_test=='Iris-setosa' , -1,1)
	X_test = df.iloc[132, [0,2] ].values
	print('Predicted Label for Testing data: '  ,AdalineModel.predict(X_test,threshold=0.6 ))
	print('True Label for Testing data : ' , y_test)


