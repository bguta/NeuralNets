import numpy as np
from sklearn.utils import shuffle

outputSize = 1
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
            # "data/basketballtrain.npy",
            # "data/light_bulbtrain.npy",
            # "data/suntrain.npy",
            # "data/cloudtrain.npy",
            # "data/eyetrain.npy",
            # "data/bicycletrain.npy",
            "data/dogtrain.npy",
            # "data/flowertrain.npy"
        ],
        [
            # "data/basketballtest.npy",
            # "data/light_bulbtest.npy",
            # "data/suntest.npy",
            # "data/cloudtest.npy",
            # "data/eyetest.npy",
            # "data/bicycletest.npy",
            "data/dogtest.npy",
            # "data/flowertest.npy"
        ]
    ]

    train_data, test_data = [], []

    for i in range(len(data[0])):
        tr = np.load(data[0][i])  # train
        ts = np.load(data[1][i])  # test

        train_data.append(tr)
        test_data.append(ts)

    train, train_goal = [], []

    for i in range(len(train_data)):
        for vct in train_data[i]:
            vec = vct.reshape(28, 28)
            train_goal.append(np.array(answers[i]))
            train.append(vec)

    train, train_goal = shuffle(train, train_goal)

    test, test_goal = [], []

    for i in range(len(test_data)):
        for vct in test_data[i]:
            vec = vct.reshape(28, 28)
            test.append(vec)
            test_goal.append(np.array(answers[i]))

    test, test_goal = shuffle(test, test_goal)

    return {"train": train, "train_goal": train_goal,
            "test": test, "test_goal": test_goal}
