import matplotlib.pyplot as plt
import numpy as np
import cv2
import os


def showlogs(logs):
    logs = np.array(logs)
    # np.save("logs.npy",logs)
    names = ["d_loss", "d_acc", "g_loss","feature_loss"]
    for i in range(4):
        plt.subplot(2, 2, i + 1)
        plt.plot(logs[:, 0], logs[:, i + 1])
        plt.xlabel("epoch")
        plt.ylabel(names[i])
    plt.tight_layout()
    plt.show()
    plt.savefig("log.jpg")

logs = np.load("logs.npy")
showlogs(logs)



