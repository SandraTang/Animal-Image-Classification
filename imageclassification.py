# classify images using python and machine learning

import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from tensorflow.keras import layers
from keras.utils import to_categorial
import numpy as numpy
import matplotlib.pyplot as plt 
plt.style.use('fivethirtyeight')

# load data
from keras.datasets import cifar10
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# look at variable data types
# all numpy arrays
print(type(x_train))
print(type(y_train))
print(type(x_test))
print(type(y_test))

# get shape of arrays
print(type('x_train shape:', x_train.shape))
print(type('y_train shape:', y_train.shape))
print(type('x_test shape:', x_test.shape))
print(type('y_test shape:', y_test.shape))

# take a look at the first image as an array
index = 0
x_train[index]

# show image as picture
img = plt.imshow(x_train[index])

# get image label
# numbers correspond with classification
print('The image label is:', y_train[index])

# get image classification
classification = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'ship', 'truck']

# print image class
# y_train[index] returns an array of size 1
print('The image class is:', classification[y_train[index][0]])

# convert labels into set of 10 numbers to input into neural network
# one hot is array of 0 except 1 where the label index is
y_train_one_hot = to_categorial(y_train)
y_test_one_hot = to_categorial(y_test)

# print new labels
print(y_train_one_hot)

# print the new label of the image above
print('The one hot label is:', y_train_one_hot[index])

# normalize pixels to be values between 0 and 1
x_train = x_train/255
x_test = x_test/255

# check new values
print(x_train[index])

# create models architecture
model = Sequential()

# add first layer
# convolution layer to extract features from input image
# outputs rectified feature map
model.add(Conv2D(32, (5, 5), activation = 'relu', input_shape = (32, 32, 3)))

# add a pooling layer
# progressively reduce spatial size of representation
# reduce amount of parameters and computation
# operates on each feature map independently
model.add(MaxPooling2D(pool_size = (2, 2)))
# max pooling 2 by 2 to get x element from feature map

# second convolution layer
model.add(Conv2D(32, (5, 5), activation = 'relu'))

# add another pooling layer 
model.add(MaxPooling2D(pool_size = (2, 2)))

# flattening later
# reduce dimensionality to linear array
model.add(Flatten())

# take all this data and add neurons
# add a layer with 1000 neurons
model.add(Dense(1000, activation = 'relu'))

# add dropout layer with 50% dropout rate
model.add(Dropout(0.5))

# add a layer with 500 neurons
model.add(Dense(500, activation = 'relu'))

# add dropout layer with 50% dropout rate
model.add(Dropout(0.5))

# add a layer with 250 neurons
model.add(Dense(250, activation = 'relu'))

# add a layer with 10 neurons
model.add(Dense(10, activation = 'softmax'))

# compile the model
model.compile(locc = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

# train the model
hist = model.fit(x_train, y_train_one_hot, match_size = 256, epochs = 10, validation_split = 0.2)

# evaluate model using test data set 
model.evaluate(x_test, y_test_one_hot)[1]

# visualize the model's accuracy
plt.plot(hist.history['accuracy'])
plt.plot(hist.history['val_accuracy'])
plt.title('Model Accuracy')
plt.x_label('Epoch')
plt.y_label('Accuracy')
plt.legend(['Train', 'Val'], loc = 'upper left')
plt.show()

# visualize the models loss
plt.plot(hist.history['loss'])
plt.plot(hist.history['val_loss'])
plt.title('Model Loss')
plt.x_label('Epoch')
plt.y_label('Loss')
plt.legend(['Train', 'Val'], loc = 'upper right')
plt.show()

# test the model with an example
from google.colab import files
uploaded = files.upload()

# upload image file

# show the image
new_image = plt.imread('image_file.jpg')
img = plt.imshow(new_image)

# image needs to be 32 by 32, so resize it
from skimage.transform import resize
resized_image = resize(new_image, (32, 32, 3))
img = plt.imshow(resized_image)

# get the models predictions
predictions = model.predict(np.array([resized_image]))

# show predictions
print(predictions)

# sort the predictions from least to greatest
list_index = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
x = predictions

for i in range(10):
	for j in range(10):
		if x[0][list_index[i]] > x[0][list_index[j]]:
			temp = list_index[i]
			list_index[i] = list_index[j]
			list_index[j] = temp

# show sorted labels in order
print(list_index)

# print first 5 predictions
for i in range(5):
	print(classification[list_index[i]], ':', round(predictions[0][list_index[i]]*100, 2), '%')



