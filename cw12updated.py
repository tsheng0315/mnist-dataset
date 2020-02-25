# cw1-part 2
# Widrow_Hoff Learning Algorithm
# margin vector is [s1....s6]--> 6 digit numbers of kcl ID
# find a linear discriminant function, to classify the data

import numpy as np
import csv

dataset_y=[[1,0,2], [1,1,2], [1,2,1], [-1,-3,1],[-1,-2,1],[-1,-3,-2]]

# initialise learning rate, margin(b), weight(a)
l_rate=0.1
b=[1,9,0,6,7,2]
a=np.array([[1],[0],[0]])

#pre-processing dataset, form y.transpose
length=len(dataset_y)
for i in range(length):
  y = dataset_y[i]
  if y[0]==-1:
    y[1]=-1*y[1]
    y[2]=-1*y[2]


# calculate parameters: margin(b), weight(a)

with open('WH.csv','w') as f_csv:
    k=0
    f_writer=csv.writer(f_csv)
    f_writer.writerow(['iteration', 'old weight', 'current feature vector y', 'g(y)', 'updated weight'])
    
    while k<12:
        bk=b[(k%6)]
        yk=dataset_y[(k%6)]
        k+=1
        aT=a.transpose()
        aTyk=aT*yk
        ykT=np.array(yk).transpose()
        a_new =aT+l_rate*(bk-aTyk)*ykT

        f_writer.writerow([k,aT[0],np.array(yk),aTyk[0],a_new[0]])

        print("iteration:",k)
        print("old weight",aT)
        print("current feature vector y",yk)
        print("g(y)",aTyk)
        print("updated weight",a_new)
        print()

