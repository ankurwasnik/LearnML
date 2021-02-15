import numpy as np
class LinearReressionGD(object):
	'''
	Requirement :
	Numpy module
	'''
	def __init__(self,lr,n_iters):
		self.lr = lr
		self.n_iters = n_iters

	def fit(self,X,y) :
		self.w_ = np.zeros(1+X.shape[1])
		self.cost_ = []

		for i in range(self.n_iters):
			y_hat = self.activation(X)
			error = (y-y_hat)
			#update rule
			self.w_[1:]+= self.lr* X.T.dot(error)
			self.w_[0] += self.lr*error.sum()
			#calculate cost
			cost = (error**2).sum()/2
			cost_.append(cost)

		return self

	def activation(self,X):
		return np.dot(X,self.w_[1:]) + self.w_[0]

	def predict(self,X):
		return self.activation(X)