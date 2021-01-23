'''
Author:Ankur Wasnik
Third Year Student (current year 2020-21) Computer Engineering

The following code is all about the learning about THE PERCEPTRON MODEL FOR CLASSIFYING IRIS DATASET. In this , I have used Perceptron model to classify only Iris-setosa data.But, you can simply use it to classify more classes.
 
'''
import pandas as pd
import numpy as np
class Perceptron (object):
	'''
	Perceptron classifier

	Attributes:
	1.learning_rate 
	2.epochs
	3.random_state

	
	'''
	def __init__(self,learning_rate=0.01,epochs=10,random_state=1):
		self.lr=learning_rate
		self.epochs = epochs
		self.random_state = random_state
	
	def fit(self,X,y):
	''' fit method is ued to train The Perceptron model  '''
		rgen=np.random.RandomState(self.random_state)
		self.w_ = rgen.normal(loc=0.0,scale=0.01, size=1+X.shape[1])
		self.errors_ = []
		for _ in range(self.epochs):
			errors=0
			for xi,target in zip(X,y):
				update=self.lr*(target-self.predict(xi))
				self.w_[0] += update #w_[0] is bias constant
				self.w_[1:] += update*xi #w_[1:] are weights 	
				errors += int(update!=0.0)
			self.errors_.append(errors)
		return self
	def net_input(self,X):
		return np.dot(X,self.w_[1:])+self.w_[0]
	def predict(self,X):
		return np.where(self.net_input(X)>=0.0 ,1,-1)
if __name__== '__main__' :
	import os 
	import pandas as pd
	s=os.path.join('https://archive.ics.uci.edu','ml','machine-learning-databases','iris','iris.data')
	print('Downloading Iris Dataset from :\n',s)
	df = pd.read_csv(s,header=None,encoding='utf-8')
	y=df.iloc[0:100,4].values
	y=np.where(y=='Iris-setosa',-1,1)
	X=df.iloc[0:100,[0,2]].values
	ppn = Perceptron(learning_rate=0.001,epochs=20)
	ppn.fit(X,y)
	import matplotlib.pyplot as plt
	plt.plot(range(1,len(ppn.errors_)+1) , ppn.errors_ , marker='o')
	plt.xlabel('Epochs')
	plt.ylabel('Number of updates')
	plt.show()	
	
