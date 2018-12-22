import model
import numpy as np
from sklearn.utils import shuffle
import random
from timeit import default_timer as timer

inputSize = 28 * 28  # the pixels space
outputSize = 8  # the number of choices for objects

labels = ["ball", "lightbulb", "sun", "cloud", "eye", "bike", "dog", "flower"]

answers = []
for i in range(outputSize):
    answers.append([0] * i + [1] + [0] * (outputSize - 1 - i))


def get_data(conv=False, doodle=True):
    """
    make the data and return a dict containg it
    """

    if doodle:

        data = [
            [
                "data/basketballtrain.npy",
                "data/light_bulbtrain.npy",
                "data/suntrain.npy",
                "data/cloudtrain.npy",
                "data/eyetrain.npy",
                "data/bicycletrain.npy",
                "data/dogtrain.npy",
                "data/flowertrain.npy"
            ],
            [
                "data/basketballtest.npy",
                "data/light_bulbtest.npy",
                "data/suntest.npy",
                "data/cloudtest.npy",
                "data/eyetest.npy",
                "data/bicycletest.npy",
                "data/dogtest.npy",
                "data/flowertest.npy"
            ]
        ]

        train_data, test_data = [], []

        # the training data
        for i in range(len(data[0])):
            tr = np.load(data[0][i])  # train
            ts = np.load(data[1][i])  # test

            train_data.append(tr)
            test_data.append(ts)

    ######################################################
        train, train_goal = [], []

        for i in range(len(train_data)):
            for vct in train_data[i]:
                if conv:
                    vec = vct.reshape(28, 28)
                else:
                    vec = vct.reshape(28 * 28, 1)
                train_goal.append(np.array(answers[i]))
                train.append(vec)

        train, train_goal = shuffle(train, train_goal)

    ######################################################
        test, test_goal = [], []

        for i in range(len(test_data)):
            for vct in test_data[i]:
                if conv:
                    vec = vct.reshape(28, 28)
                else:
                    vec = vct.reshape(28 * 28, 1)
                test.append(vec)
                test_goal.append(np.array(answers[i]))

        test, test_goal = shuffle(test, test_goal)
    else:

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
                if not conv:
                    vec = np.asfarray(vec[1:]).reshape(inputSize, 1) / 255
                else:
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
                if not conv:
                    vec = np.asfarray(vec[1:]).reshape(inputSize, 1) / 255
                else:
                    vec = np.asfarray(vec[1:]).reshape(28, 28) / 255
                test.append(vec)

    return {"train": train, "train_goal": train_goal,
            "test": test, "test_goal": test_goal}
lr = 0.001
eps = 1000
network = "config/neural_net_nodrop"


def main():

    dataset = get_data()

    data = dataset["train"]

    goal = dataset["train_goal"]

    composition = [inputSize, 500, 100,
                   outputSize]  # the network composition

    net = model.Network(composition, dropout=False)
    net.eta = lr

    # dropbox_.download(network + ".pkl", network + ".pkl")
    # dropbox_.download("config/score.txt", "config/score.txt")

    # net.load(network)  # load the network
    size = len(data)
    batch = 100000
    # print("starting...")
    count = 0
    while True:
        count += 1
        err = 0
        start = timer()
        for i in range(size):

            err += net.train(data[i], goal[i])
            prograssBar(i + 1, size)
            if (i + 1) % batch == 0:
                print("")
                print(
                    str(net.validate(dataset["test"], dataset["test_goal"])) + "%")
                net.save(network)
                # dropbox_.upload(network + ".pkl", network + ".pkl", large=True)
        searchThanConv(net, count)
        """
        with open("config/score.txt", "a") as file:
            score = net.validate(dataset["test"], dataset["test_goal"])
            file.write("\n" + str(score))
        """

        # dropbox_.upload("config/score.txt", "config/score.txt")


def searchThanConv(net, epoch, eta=lr, searchE=int(eps * 0.8), alpha=10):
    """
    search then converge- (STC) learning rate
    schedules (Darken and Moody, 1990b, 1991)

    This is constant at eta for a given number of epochs (searchE)
    than it begins to decrease

    @param alpha - a constatnt
    @pararm eta - the original learning rate
    @param searchE - the number of epochs to maintain eta for
    @param net - the neural net
    @param epoch - the epoch number

    visit here for more info: http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.42.2884&rep=rep1&type=pdf
    """
    net.eta = eta * (1 + (alpha / eta) * (epoch / searchE)) / \
        (1 + (alpha / eta) * (epoch / searchE) + (epoch**2 / searchE))


def prograssBar(val, final):
    """
    Show the prograss.

    @param val
    the current value of the prograss (you should increase this yourself)

    @param final
    the final goal
    """
    maxlen = 50
    step = final // maxlen

    print("\r[ " + "#" * (val // step) + " ] " +
          str(int(val * 100.0 / final)) + "% ", end="")


if __name__ == "__main__":
    main()
