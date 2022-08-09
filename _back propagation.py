#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
def sigmoid(n):
    res = 1 / (1 + np.exp(-n))
    return res

z = np.array([[0,0,-1],[0,1,-1],[1,0,-1],[1,1,-1],])    #Input
print(z.shape)
d = np.array([[0],[1],[1],[0],])                        #Teacher values
print("Weights V - should be ones")
v = np.random.rand(2,3)                                 #Initiate v matrix
print("value of v:\n",v)
print("Weights W - should be zeros")
w = np.random.rand(3,1)                                 #Initiate weight w
print("Initial Weight:\n",w)
print("Input:\n",z)
print("Teachers value:\n",d)
print("Initial Value of v:\n",v)
### Forward propagation
eta=1                                                    #Fix neta value
iter=7000                                           #set Iteration up to 7000
for i in range(1,iteration):
    for j,n in enumerate(z):
        #print("----Iter----",i)
        y_net = np.dot(v,z[j])                           #calculate y_net
        #print("Value of y_net:\n",y_net)
        y = sigmoid(y_net)                               #calculate y value
        y = np.append(y,[1])
        y = y.reshape(3,1)
        #print("Value of y:\n",y)
        wt = np.transpose(w)                              #Transpose weight matrix
        #print("Transpose shape:\n",wt.shape)
        out_net = np.dot(wt,y)                             #calculate out_net value
        out = sigmoid(out_net)
        #print("out value:\n",out)
        del_o =(d[j]-out)*(1-out)*out                      #calculate del_o Unipolar function
        #print("delta out:",del_o)
        #err=np.sum(del_o*w)
        del_hidden = del_o*((y)*(1-y))*w                      #Hidden Layer
        #print("del_hid value:\n",del_hid)
        w=w+eta*del_o*y                                   #Update weight
        #print("weight value:\n",w)
        del_hidden1=np.delete(del_hidden,-1)
        del_hidden1=del_hidden1.reshape(2,1)
        #print("reshape to calculate::\n",del_hid1)
        v=v+eta*del_hidden1*z[j]                             #update v value and print
        #print("The v value:\n",v)
print("Final w",w)
print("Final v",v)


# In[ ]:




