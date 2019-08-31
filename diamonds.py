#importing the diamonds dataset
import pandas as pd
df=pd.read_csv("diamonds.csv",index_col=0)
df.head()
df["cut"].unique()#unique value of the cut of diamond
#making dictionires of attributes and mapping them
cut_class_dict={"Fair": 1,"Good": 2,"Very Good": 3,"Premium": 4,"Ideal": 5}
clarity_dict={"I3":1, "I2":2, "I1":3, "SI2":4, "SI3":5, "SI1":6, "VS2":7, "VS1":8, "VVS2":9, "VVS1":10, "IF":11, "FL":12}
color_dict={"J":1, "I":2, "H":3, "G":4, "F":5, "E":6, "D":7}


df['cut']=df['cut'].map(cut_class_dict)
df['clarity']=df['clarity'].map(clarity_dict)
df['color']=df['color'].map(color_dict)

df.head()
#svm using sckit and preprocessing of data 
import sklearn
from sklearn import svm, preprocessing

df=sklearn.utils.shuffle(df)

X=df.drop("price",axis=1).values
X=preprocessing.scale(X)
Y=df['price'].values
#testing dataset
test_size=200

X_train=X[:-test_size]
Y_train=Y[:-test_size]

X_test=X[-test_size:]
Y_test=Y[-test_size:]
#comparing using both models
clf=svm.SVR(kernel="linear")
clf.fit(X_train,Y_train)
clf.score(X_test,Y_test)#0.85 accuracy

clf=svm.SVR(kernel="rbf")#0.55 accuracy
clf.fit(X_train,Y_train)
clf.score(X_test,Y_test)
for X,Y in zip(X_test,Y_test):
    print(f"Model:{clf.predict([X])[0]}, Actual:{Y}")