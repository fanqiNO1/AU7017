import os

import matplotlib.pyplot as plt
from torchvision.datasets import MNIST


def vis_mnist(dataset, width, height):
    plt.figure(figsize=(width, height))
    num = width * height
    plt.subplots_adjust(wspace=0.0, hspace=0.0)
    for i in range(num):
        plt.subplot(height, width, i + 1)
        plt.imshow(dataset[i][0], cmap='gray')
        plt.axis('off')
    this_dir = os.path.dirname(__file__)
    plt.savefig(f'{this_dir}/mnist.png', bbox_inches='tight', pad_inches=0)


def main():
    dataset = MNIST(root='./data', train=True, download=True)
    vis_mnist(dataset, 10, 5)


if __name__ == '__main__':
    main()
