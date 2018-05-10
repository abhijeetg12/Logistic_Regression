import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt


np.random.seed(420)

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

my_data = genfromtxt('q1_data_matrix.csv', delimiter=',')
dim=np.shape(my_data)
labels=  genfromtxt('q1_labels.csv', delimiter=',')
#print np.shape(labels)

max=np.zeros(dim[1])
min=np.zeros(dim[1])

for i in range(dim[1]):
    max[i]=np.max(np.transpose(my_data)[i])
    min[i]=np.min(np.transpose(my_data)[i])

X=my_data
for i in range(dim[0]):
    X[i]=(my_data[i]-min)/(max-min)


y=labels

###########################################################################


W= np.random.normal(0, 0.1,[5])#np.zeros([5,1])
b=0.5
#print np.shape(np.dot(X,W)+b)

###############################################################
##########IMPLEMENTING BATCH/ STOCHASTIC #######################

X_train=X[:700]
X_test=X[700:]


grad_w=np.zeros([5,1])
grad_b=0

grad_w_t=np.zeros([5,1])
grad_b_t=0

epochs=150
lr=1
Loss=0
lf=[]
reg=3e-4
for ep in range(epochs):
    #print W
    #print b
    Loss=0

    grad_w=np.zeros((5))
    grad_b=0
    for i in range(len(X_train)):

        grad_w_t=np.zeros(5)
        grad_b_t=0

        z=np.dot((np.transpose(W)),(X[i].reshape((5,1))))+b#(np.dot(X[i],W)+b)

        p=sigmoid(z)
        #print p

        #print p, 'probability'
        L=-y[i]*np.log((p))-(1-y[i])*np.log(1-(p))

        #print L, 'this is the loss for 1 sample'
        Loss+=L

        grad_w_t=(p-y[i])*np.transpose(X[i])+reg*W

        #grad_w_t=np.reshape(5,1)

        grad_b_t=(p-y[i])*-1

        grad_w+=grad_w_t[0]
        grad_b+=grad_b_t

        W-=lr*(1.0/1000)*grad_w_t
        b-=lr*(1.0/1000)*grad_b_t


    #print grad_w
    #print W
    # W-=lr*(1.0/1000)*grad_w
    # b-=lr*(1.0/1000)*grad_b
    lf.append(Loss)

    # print W, "W"
    # print b, "b"
    #print grad_w_t


#print W




##############################################################
#############CALCULATING ACCURACIES###########################

qy=[]
for i in (range(300)):
    z=(np.dot(X_test[i],W)+b)

    #print z
    p=sigmoid(z)
    if(p>=0.5):
        qy.append(1)
    else:
        qy.append(0)
y_test=y[700:]
qy=np.array(qy)

yz=qy-y_test

t=np.array(np.where( yz== 0))
print len(t[0])/3

#print range(len(lf))
plt.plot(range(len(lf)),lf)
plt.show()
print np.shape(lf)
print lf
print W
print b

print qy
