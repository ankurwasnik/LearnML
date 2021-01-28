import pandas as pd 
import numpy as np
from sklearn.preprocessing import LabelEncoder , StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

def load():
    df = pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data' , header=None)
    X=df.loc[:,2:].values
    y=df.loc[:,1].values
    le = LabelEncoder()
    y=le.fit_transform(y)
    Xtrain,Xtest,ytrain,ytest = train_test_split(X,y,test_size=0.3,stratify=y,random_state=69)
    return Xtrain,Xtest,ytrain,ytest

if __name__ == '__main__' :
    Xtrain , Xtest , ytrain, ytest = load()
    #make pipelines
    pipeline = make_pipeline(
        StandardScaler(),
        PCA(n_components=2),
        LogisticRegression(random_state=69)
    )
    '''
    Most of us, usually do train model like below.
    Since, we are have learned how cross validation works, we would like to implement it in real world problem solutions to get good training model that can generalize 
    Code :
    '''
    pipeline.fit(Xtrain,ytrain)
    y_pred = pipeline.predict(Xtest)
    print('Test Accuracy Without k-fold cross validation: %.3f'%(pipeline.score(Xtest,ytest)*100),'%')
    
    #cross validation : k-fold cross validation
    scores = cross_val_score(estimator=pipeline , cv=10 , X=Xtrain, y=ytrain , n_jobs=-1)
    # n_jobs = -1 means model training can be done on all cpu cores
    # cv=10 means divide data into 10 splits
    print('\nCV accuracy with k-fold cross validation: %.3f +/- %.3f '%(np.mean(scores), np.std(scores)),'%'  )