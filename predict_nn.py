import numpy as np
import sys, os
from nn import NeuralNetwork
from train_nn import load_from_pickle
from train_nn import build_dataset


def main(args):
    if args[0] == '-numbers':
        predict('mnist', '_', 10)
    elif args[0] == '-letters':
        predict('emnist-letters', '-', 26)

def predict(set, sep, num_classes):
    dataset = load_from_pickle(set, sep + "test")
    data, labels = build_dataset(dataset, num_classes)
    nn = NeuralNetwork()
    params = np.load("trained/" + set + "_params.npy")
    ans = np.argmax(labels[0])
    guess = np.argmax(nn.predict(params, data[0]))
    print("Prediction was number {}, actual number was {}".format(guess, ans))

if __name__ == '__main__':
	main(sys.argv[1:])