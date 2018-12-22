from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt


def main():
    GAN = load_model("gan.h5")  # gan.h5
    print("enter q to quit")
    while True:
        q = input("draw?")

        if q != "q":
            noise = np.random.uniform(-1, 1, (1, 100))
            pic = GAN.predict(noise).reshape(28, 28)
            show(pic)
        else:
            break


def show(pic):
    plt.imshow(pic)
    plt.show()


if __name__ == "__main__":
    main()
