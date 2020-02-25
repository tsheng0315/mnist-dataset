# part 3-2
# Neural Network- 
# Sequential Delta Learning Algorithm find weights of Linear Threshold Unit classifying dataset iris
# apply algorithm for 2 epochs
# Heaviside function H(0)=0.5
# learning rate=0.1

import numpy as np
from sklearn import datasets
iris= datasets.load_iris()

# initialise - data
datax=np.array(iris.data) # this is a 2D matrix (6*2)
classx=np.array(iris.target) # create labels  # this is a 1D vector (6*1)
l_rate=0.1

# initialise -weight
# # weight=np.array([theta,w1,w2])
theta=0
weight=[theta,-6,0,-7,2] # create initial weight list
weightOld=np.array(weight[:]) # turn weight list to array (in row)

# data augment 
# # dataset=np.concatenate((classx, datax),axis=1) # combine data x with its class label-> augmentation
# dataAug=np.insert(datax,[0],1,axis=1) # replace 0 with 1 in index colomn(axis=1)

classx=np.where(classx==0,1,0) # replace class label 0 with 1, the rest label with 0
# np.where(condition, x,y) if condition true, replace value with x, if condition false replace value with y

dataAug=np.insert(datax,0,1,axis=1) # insert 1 to 0 column(axis=1)

# define Heaviside function
def H(i):
  if i<0:
    return 0
  else:
    return 0.5

# xk=dataAug[1] # array
# xk2=xk.transpose()
# y=np.dot(weightOld,xk2)

# define Sequential Delta Learning Algorithm, return final weights
def SeqDeltaLearning(weightOld,l_rate,classx,dataAug):
  k=0
  while k<=2*len(datax):
    xkt=dataAug[k%6] # select one row from dataAug array
    classxk=classx[k%6] # no.k label in t array
    yk=np.dot(weightOld,xkt) # yk before using H()
    weightNew = weightOld + l_rate * (classxk-H(yk)) * xkt   # update weights 
    k=k+1

    weightOld= weightNew # update weight
  return weightNew

SeqDeltaLearning(weightOld,l_rate,classx,dataAug)

# use SDL function to classify Xtest
import numpy as np
import csv 
import pandas as pd


def test_data(s1,s2,s3,s4,s5,s6,s7):  # use ID to generate test data
    Stest=np.array([[s1, s2, s3, s4, s5, s6, s7],[s2, s3, s4, s5, s6, s7, s1],
    [s3, s4, s5, s6, s7, s1, s2],[s4, s5, s6, s7, s1, s2, s3]]) # 4*7 matrix
    Stest=Stest/np.array([2.3,4,1.5,4]).reshape(-1,1) # -1 => *, turn 1*4 vector to 4(-1)*1(1) 2D vector
    Xtest=Stest+np.array([4,2,1,0]).reshape(-1,1)
    Xtest=Xtest.transpose() # turn 4*7 into 7*4 matrix
    return Xtest # generate Xtest (7*4 matrix)

def dataAugment(dataset): # agument test data
  dataset=np.insert(dataset,0,1,axis=1)
  return dataset

def H(data):
  if data <0:
    return 0
  else:
    return 0.5

def Hyk(weights,data):
  valueGx = [] # g(x)
  valueH=[] # H(wx)
  with open('SDL3.csv','w') as f_csv:
      f_writer=csv.writer(f_csv)
      f_writer.writerow(['H(y)'])
  for k in range(len(data)):
    # yk=np.dot(weights,)
    xkT=np.array(data[k])
    gxk=np.dot(weights,xkT) # g(x)
    hwxk=H(gxk) # H(x)
    valueGx.append(gxk)
    valueH.append(hwxk)
    f_writer.writerow([gxk])
    k+=1
  return valueH
     


weight=SeqDeltaLearning(weightOld,l_rate,classx,dataAug)

print("SeqDeltaLearning weights:",weight)
xtest=test_data(1,9,0,6,0,7,2)  # use ID to generate test dataset
augTest=dataAugment(xtest) # agument test dataset
print("Augmented test dataset:\n",augTest)
hyk=Hyk(weight,augTest)
print(hyk)
