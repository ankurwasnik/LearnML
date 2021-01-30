import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder , StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve

def load():
    df=pd.read_csv('https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data' ,header=None)
    X = df.loc[:,2:].values
    y=df.loc[:,1].values
    lblEncoder = LabelEncoder()
    y=lblEncoder.fit_transform(y)
    return train_test_split(X,y,stratify=y,random_state=69,test_size=0.1)

def model_pipeline(random_state,max_iterations=10000):
        regressor = LogisticRegression( 
                                    penalty='l2',
                                    random_state=random_state,
                                    max_iter = max_iterations
                                )
        pipeline = make_pipeline(StandardScaler(),
                                regressor)
        return pipeline


if __name__ =="__main__" :
    print('Loading Dataset')
    Xtrain,Xtest,ytrain,ytest  = load()
    random_state=69
    print('Making Pipelines')
    pipeline = model_pipeline(random_state=random_state)
    trainsizes,trainscore,testscore = learning_curve(pipeline,X=Xtrain,y=ytrain,train_sizes=np.linspace(0.1,1.0,10),cv=10,n_jobs=-1)
    train_mean = np.mean(trainscore,axis=1)
    test_mean  = np.mean(testscore,axis=1)
    train_std  = np.std(trainscore,axis=1)
    test_std   = np.std(testscore,axis=1)
    print('Plotting learning curve')
    plt.plot(trainsizes,train_mean,label="Training accuracy")
    plt.fill_between(trainsizes,train_mean+train_std,train_mean-train_std,alpha=0.2)
    plt.plot(trainsizes,test_mean,label="Test accuracy")
    plt.fill_between(trainsizes,test_mean+test_std,test_mean-test_std,alpha=0.2)
    plt.grid()
    plt.legend()
    plt.xlabel('Number of training examples')
    plt.ylabel('Accuracy')
    plt.show()

