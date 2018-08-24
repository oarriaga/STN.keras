import matplotlib.pyplot as plt
import numpy as np


def plot_mnist_sample(mnist_sample):
    mnist_sample = np.squeeze(mnist_sample)
    plt.figure(figsize=(7, 7))
    plt.imshow(mnist_sample, cmap='gray', interpolation='none')
    plt.title('Cluttered MNIST sample', fontsize=20)
    plt.axis('off')
    plt.show()


def plot_mnist_grid(image_batch, function=None):
    fig = plt.figure()
    if function is not None:
        image_result = function([image_batch[:9]])
    else:
        image_result = np.expand_dims(image_batch[:9], 0)
    plt.clf()
    for image_arg in range(9):
        plt.subplot(3, 3, image_arg + 1)
        image = np.squeeze(image_result[0][image_arg])
        plt.imshow(image, cmap='gray')
        plt.axis('off')
    fig.canvas.draw()
    plt.show()


def print_evaluation(epoch_arg, val_score, test_score):
    message = 'Epoch: {0} | Val: {1} | Test: {2}'
    print(message.format(epoch_arg, val_score, test_score))
