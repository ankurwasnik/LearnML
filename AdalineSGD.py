import numpy as np
class AdaptiveSGD(object):

	def __init__(self,lr=0.01,epochs=100,shuffle=True,random_state=None):
		self.lr = lr
		self.epochs=epochs
		self.shuffle=shuffle
		self.random_state=random_state
		self.weight_initialized = False
	
	def initial_weights(self,m):
		self.rgen=np.random.RandomState(self.random_state)
		self.w_ = self.rgen.normal(loc=0.0,scale=0.01,size=1+m)
		self.weight_initialized = True
	
	def net_input(self,x):
		return np.dot(self.w_[1:],x)+self.w_[0]
	
	def activation(self,x):
		return x
	
	def update_weights(self,x,y):
		y_hat = self.activation(self.net_input(x))
		error = y_hat - y
		cost = np.square(np.sum(error))//2.0
		self.w_[0] += self.lr*error.sum()
		self.w_[1:] += self.lr*(np.dot(x,error))
		return cost

	def fit(self,x,y):
		self.costs = []
		self.initial_weights(x.shape[1])
		for _ in range(self.epochs):
			if self.shuffle :
				x , y = self.shuffle_(x,y)
			cost=[]
			for xi,yi in zip(x,y):
				cost.append(self.update_weights(xi,yi))
			avg_cost= sum(cost)/len(y)
			self.costs.append(avg_cost)
		return self

	def shuffle_(self,x,y):
		r = self.rgen.permutation( len(y) )
		return x[r],y[r]
	def partial_fit(self,x,y):
		if not  self.weight_initialized :
			self.initial-weights(x.shape[1])
		if y.ravel.shape[0]>1:
			#there are more than one example available for training
			for xi,yi in zip(x,y):
				self.update_weights(xi,yi)
		else :
			self.update_weights(x,y)
		return self
	
	def predict(self,x):
		return np.where(self.activation(self.net_input(x)) > 0.0 , 1, -1 )

if __name__=='__main__' :
	import os 
	import pandas as pd
	s=os.path.join('https://archive.ics.uci.edu','ml','machine-learning-databases','iris','iris.data')
	print('Downloading Iris Dataset from :\n',s)
	df = pd.read_csv(s,header=None,encoding='utf-8')
	y=df.iloc[0:100,4].values
	y=np.where(y=='Iris-setosa',-1,1)
	X=df.iloc[0:100,[0,2]].values
	learning_rate = float(input('Enter Learning Rate : '))
	epochs = int(input('Enter Epochs : '))	
	AdaptiveSGD = AdaptiveSGD(lr=learning_rate , epochs=epochs , random_state=69)
	AdaptiveSGD.fit(X,y)
	#We are done with training and You can predict on our trained model.
	''' Testing our trained data with one of trained data sample \nBut you are free to do on testing data.'''
	y_test = df.iloc[132 , 4]
	y_test = np.where(y_test=='Iris-setosa' , -1,1)
	X_test = df.iloc[132, [0,2] ].values
	print('Predicted Label for Testing data: '  ,AdaptiveSGD.predict(X_test))
	print('True Label for Testing data : ' , y_test)
		
		


	
					
