# Personalised Test Data
import numpy as np 

def test_data(s1,s2,s3,s4,s5,s6,s7):
    Stest=np.array([[s1, s2, s3, s4, s5, s6, s7],[s2, s3, s4, s5, s6, s7, s1],
    [s3, s4, s5, s6, s7, s1, s2],[s4, s5, s6, s7, s1, s2, s3]])
    Stest=Stest/np.array([2.3,4,1.5,4]).reshape(-1,1)
    Xtest=Stest+np.array([4,2,1,0]).reshape(-1,1)

    return Xtest

print(test_data(1,9,0,6,0,7,2))


