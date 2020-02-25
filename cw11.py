# Part 1: k-Nearest-Neighbour Classifier

from sklearn import datasets
#from sklearn.datasets import load_iris

from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# crate test dataset 
# def test_data(s1,s2,s3,s4,s5,s6,s7):
#     Stest=np.array([[s1, s2, s3, s4, s5, s6, s7],[s2, s3, s4, s5, s6, s7, s1],
#     [s3, s4, s5, s6, s7, s1, s2],[s4, s5, s6, s7, s1, s2, s3]])
#     Stest=Stest/np.array([2.3,4,1.5,4]).reshape(-1,1)
#     Xtest=Stest+np.array([4,2,1,0]).reshape(-1,1)

#     return Xtest

def test_data(data_list):
    Stest=np.array([data_list,data_list[1:]+data_list[:1],data_list[2:]+data_list[:2],data_list[3:]+data_list[:3]])
    Stest=Stest/np.array([2.3,4,1.5,4]).reshape(-1,1)
    Xtest=Stest+np.array([4,2,1,0]).reshape(-1,1)

    return Xtest

# print(test_data(1,9,0,6,0,7,2))

# train KNN model basd on iris dataset, then use trained model to predict label of test data
iris=datasets.load_iris()

def knn(x_train, y_train, n_neigh, x_test):
    neigh = KNeighborsClassifier(n_neighbors=n_neigh)
    neigh.fit(x_train,y_train)
    test=neigh.predict(x_test)
    return test

list =[3,7] # k=3,7
for i in list:
    if i== 3:
        print("k=3")
        print(knn(iris.data, iris.target,i,test_data([1,9,0,6,0,7,2]).transpose()))
    if i==7:
        print("k=7")
        print(knn(iris.data, iris.target,i,test_data([1,9,0,6,0,7,2]).transpose()))