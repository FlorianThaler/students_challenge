import numpy as np
from matplotlib import pyplot as plt


def visualise_data(data_arr_labeled: np.ndarray, save_figure=False):
    data_red = np.array([item[0: 2] for item in data_arr_labeled if item[2] == 1])
    data_green = np.array([item[0: 2] for item in data_arr_labeled if item[2] == 0])

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(data_red[:, 0], data_red[:, 1], c='red', s=1.2, label='class 1')
    ax.scatter(data_green[:, 0], data_green[:, 1], c='green', s=1.2, label='class 0')

    fig.legend()
    plt.show()

