# part 3-2
# Neural Network- 
# Sequential Delta Learning Algorithm find weights of Linear Threshold Unit classifying dataset iris
# apply algorithm for 2 epochs
# Heaviside function H(0)=0.5
# learning rate=0.1

import numpy as np
from sklearn import datasets
import csv
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
    with open('SDL2.csv','w') as f_csv:
        f_writer=csv.writer(f_csv)
        f_writer.writerow(['iteration', 'label', 'g(x)','new weight'])
        while k<=2*len(datax):
            xkt=dataAug[k%6] # select one row from dataAug array
            classxk=classx[k%6] # no.k label in t array
            yk=np.dot(weightOld,xkt) # yk before using H()
            weightNew = weightOld + l_rate * (classxk-H(yk)) * xkt   # update weights 
            f_writer.writerow([k,classxk,yk,weightNew])
            k=k+1
            weightOld= weightNew # update weight
    return weightNew

SeqDeltaLearning(weightOld,l_rate,classx,dataAug)
