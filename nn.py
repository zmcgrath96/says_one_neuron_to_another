import numpy as np

class NeuralNetwork:
    def __init__(self, in_nodes=784, out_nodes=10, hidden=100):
        self.in_nodes = in_nodes
        self.out_nodes = out_nodes
        self.hidden= hidden
        self.initialize_params()

    def initialize_params(self):
        # np.random.seed(0)
        self.old_settings = np.seterr(over='ignore')
        self.w1 = np.random.randn(self.hidden, self.in_nodes)
        self.w2 = np.random.randn(self.out_nodes, self.hidden)


    def train(self, data, labels, params=None):
        curr = 1
        if params is not None:
            [w1, w2] = params
            self.w1 = w1
            self.w2 = w2
        for img, label in zip(data, labels):
            self.print_status("Training", len(data), curr)
            [layer1, output] = self.forward(img)
            self.backward(layer1, output, img, label)
            curr += 1
        return [self.w1, self.w2]

    def test(self, data, labels, params=None):
        if params is not None:
            [w1, w2] = params
            self.w1 = w1
            self.w2 = w2
        total_predictions = len(data)
        correct_predictions = 0
        curr = 1
        for img, label in zip(data, labels):
             self.print_status("Testing", total_predictions, curr)
             [layer1, output] = self.forward(img)
             ans = np.argmax(label)
             predict = np.argmax(output)
             if ans == predict:
                 correct_predictions += 1
             curr += 1
        return float(correct_predictions) / total_predictions

    def predict(self, params, img):
        [w1, w2] = params
        self.w1 = w1
        self.w2 = w2
        [layer1, output] = self.forward(img)
        return output

    def print_status(self, type, total, current):
        percent_done = int(20 * float(current) / total)
        status = "[" + "|" * (percent_done) + " " * (20 - (percent_done)) + "]"
        print(type + " " + status + " {}%\t\t".format(percent_done * 5), end="\r")


    def forward(self, img):
        layer1 = self.sigmoid(np.dot(self.w1, img))
        output = self.sigmoid(np.dot(self.w2, layer1))
        return [layer1, output]

    def backward(self, layer1, output, img, label):
        diff = label - output
        d_out = 2 * diff * self.sigmoid_deriv(output)
        d_w2 = np.dot(d_out, layer1.T)

        d_l1 = np.dot(self.w2.T, d_out)
        d_w1 = np.dot(d_l1 * self.sigmoid_deriv(layer1), img.T)

        self.w1 += d_w1
        self.w2 += d_w2

    def sigmoid(self, x):
        return (1 / (1 + np.exp(-x)))

    def sigmoid_deriv(self, x):
        return x * (1.0 - x)

