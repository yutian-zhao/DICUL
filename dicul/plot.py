from matplotlib import pyplot as plt
from typing import Dict
import math
import os


def plot_stats(stats: Dict, title, ncols=3, save_path=None):
    nitems = len(stats.keys())
    # if nitems < ncols:
    #     nrows = nitems
    nrows = math.ceil(nitems / ncols)
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols*4, nrows*3), squeeze=False)
    fig.suptitle(title)
    
    for i, (k, v) in enumerate(stats.items()):
        row = i//ncols
        col = i%ncols
        axs[row][col].set_title(k)
        axs[row][col].plot([*range(len(v))], v)
        # axs[row][col].set_xlabel("Number of batches")

    fig.tight_layout()
    if save_path:
        plt.savefig(save_path)
    plt.close()

if  __name__ == "__main__":
    plot_stats({"test_data":[12,4,2,4,5]}, title="test_data", save_path="test.png")