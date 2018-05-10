import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
from sklearn import linear_model, datasets


my_data = genfromtxt('q1_data_matrix.csv', delimiter=',')
dim=np.shape(my_data)
labels=  genfromtxt('q1_labels.csv', delimiter=',')
#print np.shape(labels)
# import some data to play with

X = my_data  # we only take the first two features.
Y = labels

# step size in the mesh

logreg = linear_model.LogisticRegression(C=1)

# we create an instance of Neighbours Classifier and fit the data.
logreg.fit(X[:700], Y[:700])

yt= logreg.predict(X[700:])
kl=yt-Y[700:]


t=np.array(np.where( kl== 0))
print len(t[0])/3
#
# # Plot the decision boundary. For that, we will assign a color to each
# # point in the mesh [x_min, x_max]x[y_min, y_max].
# x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
# y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
# xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
# Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])
#
# # Put the result into a color plot
# Z = Z.reshape(xx.shape)
# plt.figure(1, figsize=(4, 3))
# plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)
#
# # Plot also the training points
# plt.scatter(X[:, 0], X[:, 1], c=Y, edgecolors='k', cmap=plt.cm.Paired)
# plt.xlabel('Sepal length')
# plt.ylabel('Sepal width')
#
# plt.xlim(xx.min(), xx.max())
# plt.ylim(yy.min(), yy.max())
# plt.xticks(())
# plt.yticks(())
#
# plt.show()
