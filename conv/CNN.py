import numpy as np

class CNN:

    def __init__(self, data, image_size, num_classes, iter=10):
        print("I'm a cnn")
        self.filter_size = 5
        self.data = data
        self.num_images = data(len)
        self.image_size = image_size
        self.num_channels = 3
        self.filter_size = 3
        self.depth = 4
        self.num_classes = num_classes
        self.learning_rate = 0.1
        self.hidden_states = 256
        self.conv_1_weights = np.random.normal(0,0.5,(self.filter_size,\
                                self.filter_size,self.num_channels,self.depth))
        self.conv_2_weights = np.random.normal(0,0.5,(self.filter_size,\
                                self.filter_size,self.depth,self.depth * 4))
        self.conv_1_biases = np.zeros([1,self.depth])
        self.conv_2_biases = np.zeros([1,self.depth * 4])
        self.full_1_weights = np.random.normal(0,0.5,(((self.image_size//4-1) * \
                                (self.image_size//4-1) * self.depth * 4),\
                                self.hidden_states))
        self.full_1_biases = np.zeros([self.hidden_states])
        self.full_2_weights = np.random.normal(0,0.5,(self.hidden_states,\
                                self.num_classes))
        self.full_2_biases = np.zeros([self.num_classes])
        self.iter = iter
        self.cached_results = dict()

    def train(self):
        print("training")
        self.forward()
        self.calculate_cost()
        self.backward()

    def forward(self, data):
        for in in range(iter):
            for img in data:
                conv1 = self.conv_layer(img, self.conv_1_weights) + self.conv_1_biases
                self.relu_layer(conv1)
                conv2 = self.conv_layer(conv1, self.conv_2_weights) + self.conv_2_biases
                self.relu_layer(conv2)
                self.max_pooling_layer()
                self.fully_conn_layer()

    def backward(self):
        # Backward propogation
        print("Backward propogation")

    def calculate_cost(self):
        # calculate_cost
        print("calculate_cost")

    def soft_max(self, arr):
        new_arr = np.exp(arr)
        sum_new_arr = np.sum(new_arr,1).reshape(-1,1)
        new_arr = new_arr/sum_new_arr
        return new_arr

    def relu_layer(self, arr):
        arr[arr<=0] = 0

    def fully_conn_layer(self):
        # fully_conn_layer
        print("fully_conn_layer")

    def conv_layer(self, image, weights):
        conv_h = (image.shape[0]-weights.shape[0])//2 + 1
        conv_w = (image.shape[1]-weights.shape[1])//2 + 1
        conv = np.zeros([conv_h,conv_w,weights.shape[3]])

        for i in range(weights.shape[3]):
            row = 0
            for j in range(0,(image.shape[0]-self.filter_size+1),2):
                col = 0
                for k in range(0,(image.shape[1]-self.filter_size+1),2):
                    conv[row,col,i] = np.sum(np.multiply(image[j:j+self.filter_size, \
                                        k:k+self.filter_size, :], weights[:,:,i]))
                    col += 1
                row += 1

        return conv

    def max_pooling_layer(self):
        # max_pooling_layer
        print("max_pooling_layer")
