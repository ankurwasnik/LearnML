from scipy.spatial.distance import pdist, squareform
from scipy import exp
from scipy.linalg import eigh
import numpy as np
from sklearn.datasets import make_moons

'''
Author: Ankur Wasnik
Date 	: Jan 24 0632
Kernel RBF(radial basis function) custom implementation
'''

def rbf_kernel_pca(X , gamma , n_components):
	#1.Calculate pairwise squared euclidean distance
	sq_dist = pdist(X,'sqeuclidean')
	#2.Convert pairwise distance into square matrix
	sq_mat = squareform(sq_dist)
	#3.Compute the kernel square matrix 
	K =np.exp(-gamma*sq_mat)
	#4.Center the kernel matrix 
	#Most important step 
	N = K.shape[0]
	one_n = np.ones((N,N))/N
	K = K -one_n.dot(K) -K.dot(one_n) + one_n.dot(K).dot(one_n)
	#5.Obtaining the eigenvalues and eigenvectors
	eigvals, eigvecs = eigh(K)
	eigvals , eigvecs = eigvals[:,:,-1] , eigvecs[:,:,-1]
	#6.Collect top K eigenvectors
	k_eigv = np.column_stack([eigvals[:,i] for i in range(n_components)])
	
	return k_eigv

'''
Implementing the kernel rbf to solve half-moon shapes 
'''
if __name__=='__main__' :
	X,y = make_moons(n_samples=100 , random_state=69)
	#1.visualizing data
	import matplotlib.pyplot as plt
	plt.scatter(X[y==0,0] , X[y==0,1])
	plt.scatter(X[y==1,0] , X[y==1,1])
	plt.tight_layout()
	plt.show()
	#2.Apply kernel rbf to data 
	X_krbf = rbf_kernel_pca(X=X , gamma=15 , n_components=2)
	#3.Visualize reduced data 
	fig,ax = plt.subplots(nrows=1,ncols=2)
	ax[0] = plt.scatter(X_krbf[y==0,0],[X_krbf[y==0,1]])	
	ax[0] = plt.scatter(X_krbf[y==1,0],[X_krbf[y==1,1]])
	ax[1] = plt.scatter(X_krbf[y==0,0],np.zeros((50,1))+0.02)
	ax[1] = plt.scatter(X_krbf[y==1,0],np.zeros((50,1))-0.02)
	ax[0].set_xlabel('PC1')
	ax[0].set_ylabel('PC2')
	ax[1].set_xlabel('PC1')
	ax[1].set_ylabel('PC2')
	plt.tight_layout()
	plt.show()

	
