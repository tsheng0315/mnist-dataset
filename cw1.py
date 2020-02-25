
import numpy as np
from sklearn import datasets

from sklearn.neighbors import KNeighborsClassifier
iris = datasets.load_iris()

s1,s2,s3,s4,s5,s6,s7=1,9,0,6,0,7,2
Stest=np.array([[s1, s2, s3, s4, s5, s6, s7],[s2, s3, s4, s5, s6, s7, s1],
[s3, s4, s5, s6, s7, s1, s2],[s4, s5, s6, s7, s1, s2, s3]]);

Stest= Stest/np.array([2.3,4,1.5,4]).reshape(-1,1)
Xtest=Stest+np.array([4,2,1,0]).reshape(-1,1)

def knn(x_train,y_train,n_neigh,x_test):
    neigh = KNeighborsClassifier(n_neighbors=n_neigh,weights='distance')
    neigh.fit(x_train,y_train)
    x_test_reshape=x_test.transpose()

    test=neigh.predict(x_test_reshape)
    #return len(x_train)
    #return x_test_reshape
    return test

print(knn(iris.data, iris.target,3,Xtest))
