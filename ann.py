import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, UpSampling2D, LeakyReLU, Input
from keras.layers import Flatten, BatchNormalization, Activation, Conv2DTranspose, Reshape
from keras.optimizers import RMSprop, Adam
from doodle_data import get
from keras.models import Model

rms = RMSprop(lr=0.0002, decay=6e-8)
rms1 = RMSprop(lr=0.0001, decay=3e-8)
adam_1 = Adam(0.0002, 0.5)
adam_2 = Adam(0.001, 0.5)
op1 = rms
op2 = rms1


def generator():
    """
    make a generator neural net;

    this generates an image from an input of noise data

    @returns the model
    """

    model = Sequential()

    in_shape = 100

    depth = 256

    model.add(Dense(depth * 7 * 7, input_shape=(in_shape,)))
    model.add(BatchNormalization(momentum=0.9))  # add the momentum
    # model.add(Activation('relu'))  # pass the vector through a relu
    model.add(LeakyReLU(alpha=0.2))

    model.add(Reshape((7, 7, depth)))  # reshape to depth number of  7x7 images
    model.add(Dropout(0.4))

    model.add(UpSampling2D())
    model.add(Conv2DTranspose(int(depth / 2), 5, padding='same'))
    model.add(BatchNormalization(momentum=0.9))
    # model.add(Activation('relu'))
    model.add(LeakyReLU(alpha=0.2))

    model.add(UpSampling2D())
    model.add(Conv2DTranspose(int(depth / 4), 5, padding='same'))
    model.add(BatchNormalization(momentum=0.9))
    # model.add(Activation('relu'))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(int(depth / 8), 5, padding='same'))
    model.add(BatchNormalization(momentum=0.9))
    # model.add(Activation('relu'))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2DTranspose(1, 5, padding='same'))
    model.add(Activation('sigmoid'))

    # model.summary()

    noise = Input(shape=(in_shape,))
    img = model(noise)

    return Model(noise, img)

    # return model


# def identifier():
#     """
#     make a discriminator neural net;

#     this determines if an image is from the dataset

#     @returns the model
#     """
#     depth = 64
#     model = Sequential()

#     model.add(Conv2D(depth, 5, strides=2, padding='same', activation=LeakyReLU(alpha=0.2),
#                      input_shape=(28, 28, 1)))

#     model.add(Dropout(0.4))

#     model.add(Conv2D(depth * 2, 5, strides=2, padding='same',
#                      activation=LeakyReLU(alpha=0.2)))
#     model.add(Dropout(0.4))

#     model.add(Conv2D(depth * 4, 5, strides=2, padding='same',
#                      activation=LeakyReLU(alpha=0.2)))

#     model.add(Dropout(0.4))

#     model.add(Conv2D(depth * 8, 5, strides=2, padding='same',
#                      activation=LeakyReLU(alpha=0.2)))

#     model.add(Dropout(0.4))

#     model.add(Flatten())

#     model.add(Dense(1, activation="sigmoid"))

#     # model.summary()

#     img = Input(shape=(28, 28, 1))
#     validity = model(img)

#     return Model(img, validity)

#     # return model


################################
# the inspector model
depth = 64
model = Sequential()

model.add(Conv2D(depth, 5, strides=2, padding='same', input_shape=(28, 28, 1)))
model.add(LeakyReLU(alpha=0.2))
model.add(Dropout(0.4))

model.add(Conv2D(depth * 2, 5, strides=2, padding='same'))
model.add(LeakyReLU(alpha=0.2))
model.add(Dropout(0.4))

model.add(Conv2D(depth * 4, 5, strides=2, padding='same'))
model.add(LeakyReLU(alpha=0.2))
model.add(Dropout(0.4))

model.add(Conv2D(depth * 8, 5, strides=2, padding='same'))
model.add(LeakyReLU(alpha=0.2))
model.add(Dropout(0.4))

model.add(Flatten())

model.add(Dense(1, activation="sigmoid"))

# model.summary()

img = Input(shape=(28, 28, 1))
validity = model(img)

################################


def discriminator():
    """
    compile the identifier model to be used and return it
    """

    # img = Input(shape=(28, 28, 1))
    # validity = ident(img)

    model = Model(img, validity)

    model.compile(loss="binary_crossentropy", optimizer=op1,
                  metrics=['accuracy'])

    # model.summary()

    return model


def gan(ident=None, gan=None):
    """
    make a generative adversarial network using the
    identifier and the generator return it
    """

    model = Sequential()

    model.add(gan)
    model.add(ident)

    model.compile(loss='binary_crossentropy', optimizer=op2,
                  metrics=['accuracy'])

    # model.summary()

    return model


def train(epochs=100, batch=30, info=5000):
    """
        train the gan and the discriminator to create a working generator
    """
    data = get()
    info = batch * 200
    batch_backup = batch

    images = np.concatenate((data["train"], data["test"]))

    images = images.reshape(len(images), 28, 28, 1)
    dataSize = len(images)

    print("data size:", dataSize)
    print("Batch size:", batch)

    dis = discriminator()

    ident = Model(img, validity)
    ident.trainable = False
    ident.compile(loss='binary_crossentropy', optimizer=op1,
                  metrics=['accuracy'])

    gener = generator()
    gener.compile(loss='binary_crossentropy', optimizer=op2,
                  metrics=['accuracy'])

    adversarial = gan(gan=gener, ident=ident)

    for epoch in range(epochs):
        batch = batch_backup
        start = 0
        end = batch
        while start <= dataSize:

            if end >= dataSize:
                real = images[start:]
                batch = dataSize % batch
            else:
                real = images[start:end]

            noise = np.random.uniform(-1, 1, (batch, 100))
            fake = gener.predict(noise)

            x = np.concatenate((real, fake))

            y = np.ones((2 * batch, 1))
            y[batch:, :] = 0

            disLoss = dis.train_on_batch(x, y)

            y = np.ones((batch, 1))

            #noise = np.random.uniform(-1, 1, (batch, 100))

            advLoss = adversarial.train_on_batch(noise, y)

            n = np.random.uniform(-1, 1, (1, 100))
            p = gener.predict(n)

            prograssBar(start, dataSize)

            if end % (info * 2) == 0:
                print("")
                gener.save("Dog2GAN.h5")
                print("saved: {}".format(epoch))

            if end % info == 0:
                stats = "D: {} ; G: {}".format(disLoss, advLoss)
                print(stats)

            if end >= dataSize:
                break

            start += batch
            end += batch


def prograssBar(val, final):
    """
    Show the prograss.

    @param val
    the current value of the prograss (you should increase this yourself)

    @param final
    the final goal
    """
    end = ""
    maxlen = 50
    step = final // maxlen

    print("\r[ " + "#" * (val // step) + " ] " +
          str(int(val * 100.0 / final)) + "% ", end=end)

if __name__ == "__main__":
    train()
