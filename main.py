import numpy as np
import matplotlib.image as image
import formatImage
from keras.models import load_model
from draw import show

# 96.55 % accuracy on doodle
labels = ["ball", "lightbulb", "sun", "cloud", "eye", "bike", "dog", "flower"]

# 98.86 % accuracy on mnist


def main():
    choice = int(input("please enter 1 for minst or 0 for doodle network: "))

    if choice == 1:
        net = load_model("mnist.h5")
    else:
        net = load_model("doodle.h5")

    """
    GAN = load_model("gan.h5")

    while True:
        q = input("draw?")

        if q != "q":
            noise = np.random.uniform(-1, 1, (1, 100))
            pic = GAN.predict(noise)
            v = net.predict(pic)
            print(np.argmax(v))
            show(pic.reshape(28, 28))
        else:
            break
    """

    while True:
        pic = input("please input the name of the file ('q' to exit): ")
        if pic == "q":
            break
        else:
            test(pic, net, choice=choice)


def test(img, nn, choice=1):
    """
    Test an image with the network
    """
    if ".png" in img:
        try:
            formatImage.format(img, invert=True)
            pic = image.imread(img)

            pixels = pic.reshape(1, 28, 28, 1)

            v = nn.predict(pixels)
            # print(str(v))
            if choice == 1:
                print(np.argmax(v))
            else:
                ans = np.argmax(v)
                print(labels[ans])

        except Exception as e:
            print(e)
            pass

if __name__ == "__main__":
    main()
