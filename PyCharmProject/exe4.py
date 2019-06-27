
# from sklearn.cluster import KMeans
# from __future__ import print_function
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.spatial.distance import cdist
# import random
# np.random.seed(18)
# means = [[2, 2], [8, 3], [3, 6]]
# cov = [[1, 0], [0, 1]]
# N = 500
# X0 = np.random.multivariate_normal(means[0], cov, N)
# X1 = np.random.multivariate_normal(means[1], cov, N)
# X2 = np.random.multivariate_normal(means[2], cov, N)
# X = np.concatenate((X0, X1, X2), axis = 0)
# K = 3
# original_label = np.asarray([0]*N + [1]*N + [2]*N).T
# model = KMeans(n_clusters= = 3, random_state = 0).fit(X)

# import matplotlib.image as mpimg
# import matplotlib.pyplot as plt
# import numpy as np
# from sklearn.cluster import KMeans
# img = mpimg.imread('girl.jpg')
# plt.imshow(img)
# imgplot = plt.imshow(img)
# plt.axis('off')
# plt.show()
# X = img.reshape((img.shape[0]*img.shape[1], img.shape[2]))
# for k in [3, 5, 10, 15, 20]:
#     kmeans = KMeans(n_clusters = k).fit(X)
#     label = kmeans.predict(X)
#     img4 = np.zeros_like(X)
#     for k in range(k):
#         img4[label == k] = kmeans.cluster_centers_[k]
#         img5 = img4.reshape((img.shape[0],img.shape[1], img.shape[2]))
#         plt.imshow(img5, interpolation='nearest')
#         plt.axis('off')
#         plt.show()


# import numpy as np
# def predict(w,X):
#     return np.sign(X.dot(w))
#
# def percep(X, y, w_init):
#     w = w_init
#     while True:
#         pred = predict(w, X)
#         mis_idxs = np.where(np.equal(pred, y) == False)[0]
#         num_mis = mis_idxs.shape[0]
#         if num_mis == 0:
#             return  w
#         random_id = np.random.choice(mis_idxs, 1)[0]
#         w = w +y[random_id]*X[random_id]

from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

batch_size = 128
num_class = 10
epochs = 10
img_row, img_col = 28, 28

(xtrain, ytrain), (xtest, ytest) = mnist.load_data()
if K.image_data_format() == 'channels_first':
    xtrain = xtrain.reshape(xtrain.shape[0], 1, img_row, img_col)
    xtest = xtest.reshape(xtest.shape[0],1, img_row, img_col)
    input_shape = (1, img_row, img_col)
else:
    xtrain = xtrain.reshape(xtrain.shape[0], img_row, img_col,1)
    xtest = xtest.reshape(xtest.shape[0], img_row, img_col, 1)
    input_shape = (img_row, img_col, 1)

xtrain = xtrain.astype('float32')
xtest = xtest.astype('float32')
xtrain = xtrain/255
xtest = xtest /255
print(xtrain.shape)
print(xtest.shape)
ytrain = keras.utils.to_categorical(ytrain, num_class)
ytest = keras.utils.to_categorical(ytest, num_class)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_class, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(xtrain, ytrain,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(xtest, ytest))
scoretrain = model.evaluate(xtrain, ytrain, verbose=0)
print('Train loss: ',scoretrain[0])
print('Train accuracy: ', scoretrain[1])
scoretest = model.evaluate(xtest, ytest, verbose=0)
print('Test loss:', scoretest[0])
print('Test accuracy:', scoretest[1])