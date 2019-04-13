import numpy as np

class CNN:

    def __init__(self, data, image_size, num_classes, batch_size=0):
        print("I'm a cnn")
        self.filter_size = 5
        if batch_size == 0:
            self.batch_size = len(data)
        else:
            self.batch_size = batch_size
        self.image_size = image_size
        self.num_channels = 3
        self.filter_size = 3
        self.depth = 4
        self.num_classes = num_classes
        self.learning_rate = 0.1
        self.hidden_states = 256
        self.conv_1_weights = np.random.normal(0,0.5,self.filter_size,\
                                self.filter_size,self.num_channels,self.depth)
        self.conv_2_weights = np.random.normal(0,0.5,self.filter_size,\
                                self.filter_size,self.depth,self.depth * 4)
        self.conv_1_biases = np.zeros([1,self.depth])
        self.conv_2_biases = np.zeros([1,self.depth * 4])
        self.full_1_weights = np.random.normal(0,0.5,(((self.image_size//4-1) * \
                                (self.image_size//4-1) * self.depth * 4),\
                                self.hidden_states))
        self.full_1_biases = np.zeros([self.hidden_states])
        self.full_2_weights = np.random.normal(0,0.5,(self.hidden_states,\
                                self.num_classes))
        self.full_2_biases = np.zeros([self.num_classes])

    def train(self):
        print("training")
        self.forward()
        self.calculate_cost()
        self.backward()

    def forward(self):
        # Forward propagation
        print("Forward propagation")
        self.conv_layer_1()
        self.relu_layer()
        self.conv_layer_2()
        self.relu_layer()
        self.max_pooling_layer()
        self.fully_conn_layer()

    def backward(self):
        # Backward propogation
        print("Backward propogation")

    def calculate_cost(self):
        # calculate_cost
        print("calculate_cost")

    def soft_max(self):
        # soft max
        print("soft max")

    def relu_layer(self):
        # relu_layer
        print("relu_layer")

    def fully_conn_layer(self):
        # fully_conn_layer
        print("fully_conn_layer")

    def conv_layer_1(self):
        # conv_layer
        print("conv_layer_1")

    def conv_layer_2(self):
        # conv_layer
        print("conv_layer_2")

    def max_pooling_layer(self):
        # max_pooling_layer
        print("max_pooling_layer")
