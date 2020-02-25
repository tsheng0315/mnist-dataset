# part 3-1
# Neural Network- Sequential Delta Learning Algorithm
# find weights of Linear Threshold Unit 
# to classify dataset
# until convergence to solution or until 12 iterations.
# Heaviside function H(0)=0.5
# learning rate=1

import numpy as np
import csv
# initialise - data
datax=np.array([[0,2], [1,2], [2,1], [-3,1],[-2,1],[-3,-2]]) # this is a 2D matrix (6*2)
classx=np.array([[1],[1],[1],[0],[0],[0]])  # create labels  # this is a 2D matrix (6*1)

# initialise - learning rate
l_rate=0.1

# initialise - weight
# # w1,w2=0,0
# # weight=np.array([theta,w1,w2])

theta=0
# theta =1
weight=[theta,-6,0] # list
#weight=[theta,-0,0] # list
weightOld=np.array(weight[:])# array in row

# data augment 
# #dataset=np.concatenate((classx, datax),axis=1) # combine data x with its class label-> augmentation
dataAug=np.insert(datax,0,1,axis=1) # replace 0 with 1 in index colomn(axis=1)

# define Heaviside function
def H(i):
  if i<0:
    return 0
  else:
    #return 1
    return 0.5

# xk=dataAug[1] # array
# xk2=xk.transpose()
# y=np.dot(weightOld,xk2)

def SeqDeltaLearning(weightOld,l_rate,classx,dataAug): 
    k=0
    with open('WH.csv','w') as f_csv:
        f_writer=csv.writer(f_csv)
        f_writer.writerow(['iteration', 'old weight', 'X.T', 'g(x)', 'H(wx)','label','new weight'])
        while k<=11:
            xtk=dataAug[k%6] # select one row from dataAug array
            classxk=classx[k%6] # tk array
            yk=np.dot(weightOld,xtk) # yk before using H()
            weightNew = weightOld + l_rate * (classxk-H(yk)) * xtk
            f_writer.writerow([k,weightOld,xtk,yk,H(yk),classxk,weightNew])
            k=k+1 
            
    #         print("iteration:",k) 
    #   print("old weight:",weightOld)
    #   print("X.T;",xtk)
    #   print("g(x):", yk)
    #   print("H(wx):",H(yk))
    #   print("label:",classxk)
    #   print("new weight:",weightNew)
    #   print()  
    
            weightOld=weightNew
            
    return 


SeqDeltaLearning(weightOld,l_rate,classx,dataAug)