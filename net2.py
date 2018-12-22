from network import get_data
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD
import numpy as np


def main():
    # the data
    data = get_data(conv=True, doodle=False)

    train = np.array(data["train"])
    train = train.reshape(len(data["train"]), 28, 28, 1)
    goal = np.array(data["train_goal"])

    test = np.array(data["test"])

    test = test.reshape(len(data["test"]), 28, 28, 1)
    test_goal = np.array(data["test_goal"])

################################################
    # the model
    model = Sequential()

    model.add(Conv2D(32, (3, 3), activation='relu',
                     input_shape=(28, 28, 1)))
    """
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
    """
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(200, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
################################################

    sgd = SGD(lr=0.001, decay=1e-6, momentum=0.9, nesterov=True)

    # model.compile(loss='mean_squared_error', optimizer=sgd,
    #               metrics=['accuracy'])

    model.compile(loss='categorical_crossentropy',
                  optimizer='Adadelta',
                  metrics=['accuracy'])
    print("Ready")

    model.fit(train, goal, epochs=4, batch_size=128, verbose=1,
              validation_data=(test, test_goal))

    print("done")

    while True:
        name = input("Please enter the file name to save to (without .h5): ")
        try:
            model.save(name + ".h5")
            break
        except Exception as e:
            print(str(e))

    print("saved")

if __name__ == "__main__":
    main()
