import tensorflow as tf
import matplotlib.pyplot as plt

from keras.datasets import cifar10
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Flatten

from sklearn.metrics import classification_report

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# plt.imshow(x_train[1])
# plt.show()

# normalizing the data
y_categorical_train = to_categorical(y_train, 10)
y_categorical_test = to_categorical(y_test, 10)
x_train = x_train / 225
x_test = x_test / 255

# building the model
model = Sequential()

# first set
# convolutional layer
model.add(Conv2D(filters=32, kernel_size=(4, 4), input_shape=(32, 32, 3), activation='relu', ))
# pooling layer
model.add(MaxPool2D(pool_size=(2, 2)))

# second set
# convolutional layer
model.add(Conv2D(filters=32, kernel_size=(4, 4), input_shape=(32, 32, 3), activation='relu', ))
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
model.fit(x_train, y_categorical_train, verbose=1, epochs=10)

# model evaluation
model.evaluate(x_test, y_categorical_test)
predictions = model.predict_classes(x_test)
print(classification_report(y_test, predictions))

# model.save('my_model.h5')

# building the larger model
model = Sequential()

# first set
model.add(Conv2D(filters=32, kernel_size=(4, 4), input_shape=(32, 32, 3), activation='relu', ))
model.add(Conv2D(filters=32, kernel_size=(4, 4), input_shape=(32, 32, 3), activation='relu', ))
model.add(MaxPool2D(pool_size=(2, 2)))

# second set
model.add(Conv2D(filters=64, kernel_size=(4, 4), input_shape=(32, 32, 3), activation='relu', ))
model.add(Conv2D(filters=64, kernel_size=(4, 4), input_shape=(32, 32, 3), activation='relu', ))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(10, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.fit(x_train, y_categorical_train, verbose=1, epochs=20)
model.evaluate(x_test, y_categorical_test)
predictions = model.predict_classes(x_test)
print(classification_report(y_test, predictions))

# model.save('my_model2.h5')
