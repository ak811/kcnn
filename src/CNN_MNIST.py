import tensorflow as tf
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten

from sklearn.metrics import classification_report

(x_train, y_train), (x_test, y_test) = mnist.load_data()
single_image = x_train[0]
# plt.imshow(single_image)
# plt.show()

# normalizing the data
y_categorical = to_categorical(y_train)
y_categorical_test = to_categorical(y_test, 10)
y_categorical_train = to_categorical(y_train, 10)
x_train = x_train / 255
x_test = x_test / 255
# plt.imshow(x_train[0])
# plt.show()

# reshape to include channel dimension
x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

# building the model
model = Sequential()
# convolutional layer
model.add(Conv2D(filters=32, kernel_size=(4, 4), input_shape=(28, 28, 1), activation='relu', ))
# pooling layer
model.add(MaxPool2D(pool_size=(2, 2)))
# image flattening => 28 by 28 = 784
model.add(Flatten())
# 128 neurons in dense hidden layer
model.add(Dense(128, activation='relu'))
# 10 possible classes for the last layer (classifier)
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
model.summary()

# model training
model.fit(x_train, y_categorical_train, epochs=2)

# model evaluation
model.evaluate(x_test, y_categorical_test)
predictions = model.predict_classes(x_test)
print(classification_report(y_test, predictions))

# model.save('my_model.h5')
