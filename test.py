import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, UpSampling2D, LeakyReLU, Input
from keras.layers import Flatten, BatchNormalization, Activation, Conv2DTranspose, Reshape
from keras.optimizers import RMSprop
from mnist_data import get as mnist
from keras.models import Model

rms = RMSprop(lr=0.0004, clipvalue=1.0, decay=3e-8)

"""
Testing to see how keras freezes the weights

"""

model = Sequential()

model.add(Dense(100, activation="relu", input_shape=(10,)))
model.add(Dense(1, activation="sigmoid"))

noise = Input(shape=(10,))
validity = model(noise)

Model(noise, validity)


model = Model(noise, validity)
model.compile(optimizer='rmsprop', loss='mse')

model2 = Model(noise, validity)
model2.trainable = False
model2.compile(optimizer='rmsprop', loss='mse')

x = np.random.random(size=(100, 10))
y = np.random.random((100, 1))

print("First")
model.fit(x, y, epochs=4)
print("Second")
model2.fit(x, y, epochs=4)
print("First")
model.fit(x, y, epochs=4)
print("Second")
model2.fit(x, y, epochs=4)


v = np.random.random((1, 10))
print(model.predict(v))
print(model2.predict(v))
