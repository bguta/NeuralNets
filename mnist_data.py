import numpy as np
import random

outputSize = 10
answers = []

for i in range(outputSize):
    answers.append([0] * i + [1] + [0] * (outputSize - 1 - i))


def get():
    """
    make the data and return a dict containg it. The dict is in the form:

    {"train": <>, "train_goal": <>, "test": <>, "test_goal": <>}

    the data is a 28 x 28 normalized between (0-1) numpy array
    """

    data = [
        [
            "data/mnist_train.csv",
        ],
        [
            "data/mnist_test.csv",
        ]
    ]

    train = []
    train_goal = []

    with open(data[0][0], "r") as file:
        info = file.readlines()
        random.shuffle(info)

        for line in info:
            vec = line.split(",")
            train_goal.append(answers[int(vec[0])])

            vec = np.asfarray(vec[1:]).reshape(28, 28) / 255
            train.append(vec)

    test = []
    test_goal = []

    with open(data[1][0], "r") as file:
        info = file.readlines()
        random.shuffle(info)

        for line in info:
            vec = line.split(",")
            test_goal.append(answers[int(vec[0])])
            vec = np.asfarray(vec[1:]).reshape(28, 28) / 255
            test.append(vec)

    return {"train": train, "train_goal": train_goal,
            "test": test, "test_goal": test_goal}
