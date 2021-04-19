import pandas as pd 
from sklearn.preprocessing import LabelEncoder , StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression

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
    pipeline.fit(Xtrain,ytrain)
    y_pred = pipeline.predict(Xtest)
    print('Test Accuracy: %.3f'%(pipeline.score(Xtest,ytest)*100),'%')