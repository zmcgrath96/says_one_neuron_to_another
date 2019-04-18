import numpy as np
from scipy.special import softmax
import sys

class CNN:

    def __init__(self):
        self.num_channels = 3
        self.filter_size = 3
        self.depth = 4
        self.regularization = 0
        self.learning_rate = 0.1
        self.hidden_states = 256
        #AKA filters for the 2 convolutions
        self.conv_1_weights = None
        self.conv_2_weights = None
        self.conv_1_biases = None
        self.conv_2_biases = None
        self.full_1_weights = None
        self.full_1_biases = None
        self.full_2_weights = None
        self.full_2_biases = None
        self.iter = None
        self.cached_results = None
        self.labels = None

    '''
    Description:
        Go through the entire set 'iter' number of Epochs. Batch the data
        into the the batch_size and make that many forward passes. Make the
        backward pass after that for that set and adjust weights

    Parameters:
        data: training data set [num_images][image_size][image_size][rgb]
        image_size: n number of pixels of an nxn image (numpy array)
        labels: label for the images (one hot array): [num_images][num_classes]
        iter: number of epochs to train over
        batch_size: number of images per forward/backword run
    '''
    def train(self, data, image_size, labels, iter=1, batch_size=20):
        self.initialize_params(data, image_size, labels, iter)
        for it in range(iter):
            print('Epoch {}'.format(it))
            for step in range(0, len(data), batch_size):
                #get the batch data.
                start = step
                end = start + batch_size

                batch_data = data[start:end,:,:,:]
                batch_labels = labels[start:end]

                print('Batch {}'.format(step))

                output = self.forward(batch_data)
                loss, accuracy = self.calculate_cost(batch_labels, output)
                derivatives = self.backward(batch_data, batch_labels)
                self.update_parameters(derivatives)

                # print loss and accuracy of the batch dataset.
                # if(step%10==0):
                #     print('Step : %d'%step)
                #     print('Loss : %f'%loss)
                #     print('Accuracy : %f%%'%(round(accuracy*100,2)))

    def initialize_params(self, data, image_size, labels, num_classes):
        self.data = data
        self.num_images = len(data)
        self.image_size = image_size
        self.num_classes = num_classes
        self.conv_1_weights = np.random.normal(0,0.5,(self.filter_size,\
                                self.filter_size,self.num_channels,self.depth))
        self.conv_2_weights = np.random.normal(0,0.5,(self.filter_size,\
                                self.filter_size, self.depth, self.depth * 4))
        self.conv_1_biases = np.zeros([1,self.depth])
        self.conv_2_biases = np.zeros([1,self.depth * 4])
        self.full_1_weights = np.random.normal(0,0.5,(((self.image_size//4-1) * \
                                (self.image_size//4-1) * self.depth * 4),\
                                self.hidden_states))
        self.full_1_biases = np.zeros([self.hidden_states])
        self.full_2_weights = np.random.normal(0,0.5,(self.hidden_states,\
                                self.num_classes))
        self.full_2_biases = np.zeros([self.num_classes])
        self.cached_results = dict()
        self.labels = labels

    '''
    Description:
        Forward pass of the CNN. Predicts what the image is based on the
        images passed in

    Parameters:
        data: [batch_length] images to train on

    Returns:
         output: [num_classes] of likelihood of class
    '''
    def forward(self, data):
        # lists to hold convolutions of each image in the batch
        conv1_list = list()
        conv2_list = list()

        for img in data:
            # convolute each layer and squash them with relu
            conv1 = self.forward_conv(img, self.conv_1_weights, self.conv_1_biases)
            self.relu_layer(conv1)

            conv2 = self.forward_conv(conv1, self.conv_2_weights, self.conv_2_biases)
            self.relu_layer(conv2)

            conv2 = conv2.reshape((self.image_size // 4 - 1) * (self.image_size // 4 - 1)\
                    * self.depth * 4)
            conv1_list.append(conv1)
            conv2_list.append(conv2)

        conv1_arr = np.array(conv1_list).reshape(len(data), self.image_size // \
                        2 - 1, self.image_size // 2 - 1, self.depth)
        conv2_arr = np.array(conv2_list).reshape(len(data), self.image_size // \
                        4 - 1, self.image_size // 4 - 1, self.depth * 4)

        # Max pooling layer
        pooled = np.zeros(conv2_arr.shape[0])
        stride = 4
        pooling_width = 4
        for img in range(conv2_arr.shape[0]):
            pooled[img] = self.max_pool(conv2_arr[2], pooling_width, stride)

        # Flatten the data 
        arr = self.expand(pooled, len(data))
        full1 = self.fully_conn_layer(arr, self.full_1_weights, self.full_1_biases)
        self.relu_layer(full1)
        full2 = self.fully_conn_layer(full1, self.full_2_weights, self.full_2_biases)
        output = softmax(full2)

        # Cache the results
        self.cached_results['conv1'] = conv1_arr
        self.cached_results['conv2'] = conv2_arr
        self.cached_results['pooled'] = pooled
        self.cached_results['full1'] = full1
        self.cached_results['full2'] = full2
        self.cached_results['output'] = output
        return output

    '''
    Description:
        Reverse pass on the CNN. Adjusts weights/values of convolution and fully
        connected layer to improve prediction

    Parameters:
        data: [batch_size] of images (numpy arrays)
        labels: [batch_size] of [num_classes] one hot arrays of classes

    Returns:
        deriv: {} of derivatives for each convolution and each fully connected layer
    '''
    def backward(self, data, labels):
        conv1 = self.cached_results['conv1']
        conv2 = self.cached_results['conv2']
        full1 = self.cached_results['full1']
        full2 = self.cached_results['full2']
        output = self.cached_results['output']

        derivs_out = output - labels

        #loss gradient of full1
        derivs_weights_f = derivs_out.dot(full1.T)
        # loss gradient for the bias
        derivs_bias = np.sum(derivs_weights_f, 1).reshape(full2.shape)
        
        ##### TODO: the rest of this. Not really sure whats going on yet

        temp_arr = np.array(conv2).reshape(-1,(self.image_size // 4-1) * \
                    (self.image_size // 4 -1)* self.depth * 4)

        error_full2 = output - labels
        error_full1 = np.matmul(error_full2, self.full_2_weights.T)
        error_full1 = np.multiply(error_full1, full1)
        error_full1 = np.multiply(error_full1, (1 - full1))

        error_t = np.multiply(np.multiply(np.matmul(error_full1, self.full_1_weights.T), \
                    temp_arr), (1 - temp_arr))

        n = data.shape[0]

        error_conv2 = np.array(error_t).reshape(-1, self.image_size // 4 - 1, \
                        self.image_size // 4 - 1, self.depth * 4)
        error_conv1 = np.zeros(conv1.shape)

        for i in range(n):
            error = self.get_conv_errors(error_conv2[i], self.conv_2_weights)
            error = np.multiply(error, conv1[i])
            error = np.multiply(error, (1 - conv1[i]))
            error_conv1 += error

        deriv_full2 = (np.matmul(full1.T, error_full2) + self.regularization * \
                        self.full_2_weights) / n
        deriv_full1 = (np.matmul(temp_arr.T, error_full1) + self.regularization * \
                        self.full_1_weights) / n

        deriv_conv2 = np.zeros(self.conv_2_weights.shape)
        deriv_conv1 = np.zeros(self.conv_1_weights.shape)
        for i in range(n):
            deriv_conv2 = deriv_conv2 + self.get_deriviatives(error_conv2[i], conv1[i])
            deriv_conv1 = deriv_conv1 + self.get_deriviatives(error_conv1[i], data[i])
            deriv_conv2 = (deriv_conv2 + self.regularization * self.conv_2_weights) / n
            deriv_conv1 = (deriv_conv1 + self.regularization * self.conv_1_weights) / n

        deriv = dict()

        deriv['deriv_conv1'] = deriv_conv1
        deriv['deriv_conv2'] = deriv_conv2
        deriv['deriv_full1'] = deriv_full1
        deriv['deriv_full2'] = deriv_full2

        return deriv

    def forward_conv(self, image, weights, bias):
        conv_h = (image.shape[0] - weights.shape[0]) // 2 + 1
        conv_w = (image.shape[1] - weights.shape[1]) // 2 + 1
        conv = np.zeros([conv_h, conv_w, weights.shape[3]])

        for i in range(weights.shape[3]):
            row = 0
            for j in range(0, (image.shape[0] - self.filter_size + 1), 2):
                col = 0
                for k in range(0, (image.shape[1] - self.filter_size + 1), 2):
                    conv[row,col,i] = np.sum(image[j:j+self.filter_size, \
                                        k:k+self.filter_size, :] * weights[:,:,:,i])
                    col += 1
                row += 1

        return conv + bias

    def reverse_conv(self, derivs_previous, curr_conv, filt, stride):
        derivs_out = np.zeros(curr_conv.shape)
        derivs_filt = np.zeros(filt.shape)
        derivs_bias = 0

        (f_width, f_height, _) = filt.shape
        (_, conv_height, _) = curr_conv.shape

        curr_y = small_y = 0
        while curr_y + f_height <= conv_height:
            curr_x = small_x = 0
            while curr_x + f_width < conv_height:
                derivs_filt += derivs_previous[small_y, small_x] * curr_conv[curr_y:curr_y + f_height, curr_x:curr_x + f_width, :]
                derivs_out[curr_y:curr_y + f_height, curr_x:curr_x + f_width, :] = derivs_previous[small_y, small_x] * filt
                small_x += 1
                curr_x += stride
            small_y += 1
            curr_y += stride
        
        derivs_bias = np.sum(derivs_previous)
        return derivs_out, derivs_filt, derivs_bias


    '''
    Description:
        Using a width of the pool and the stride length, take the max value for every frame in the
        image of the width. There is no max done for the depth of the image

    Parameters:
        img: image to pool
        frame_w: the width of the pooling window
        stride: how far to move the pooling window

    Returns:
        pooled: the pooled version of the image passed in
    '''
    def max_pool(self, img, frame_w, stride):
        (old_width, old_height, depth) = img.shape

        # max pooled numpy array will have size (reduced height, reduced width, depth)
        new_width = ((old_width - frame_w)/stride) + 1
        new_height = ((old_height - frame_w)/stride) + 1

        # the pooled numpy array
        pooled = np.zeros((new_width, new_height, depth))

        # go through the image vertically
        curr_y = pooled_x = 0
        while curr_y + frame_w <= old_height:
            # go through the image horizontally
            curr_x = pooled_y = 0
            while curr_x + frame_w <= old_width:
                # go through the depth of the image
                for c in range(depth):
                    pooled[pooled_y, pooled_x, c] = np.max(img[curr_y+frame_w, curr_x+frame_w, c])

                curr_x += stride 
                pooled_x += 1
            curr_y += stride 
            pooled_y += 1

        return pooled


    def get_conv_errors(self, n_error, weight):
        errors = np.zeros([n_error.shape[0] * 2 + 2, n_error.shape[1] * 2 + 2, \
                    n_error.shape[2] // 4])
        for i in range(weight.shape[3]):
            row = 0
            for j in range(0, errors.shape[0] - self.filter_size + 1, 2):
                col=0
                for k in range(0, errors.shape[2] - self.filter_size + 1, 2):
                    errors[j:j + self.filter_size, k:k + self.filter_size,:] \
                        += weight[:,:,:,i] * n_error[row,col,i]
                    col+=1
                row +=1
        return errors

    def get_deriviatives(self, errors, conv_arr):
        derivatives = np.zeros([self.filter_size, self.filter_size, \
                        conv_arr.shape[2], errors.shape[2]])
        for i in range(0, derivatives.shape[3]):
            row=0
            for j in range(0, conv_arr.shape[0] - self.filter_size + 1, 2):
                col = 0
                for k in range(0, conv_arr.shape[1] - self.filter_size + 1, 2):
                    derivatives[:,:,:,i] += np.multiply(conv_arr[j:j + self.filter_size,\
                                k:k + self.filter_size, :], errors[row,col,i])
                    col += 1
                row += 1
        return derivatives

    def apply_derivatives(self, deriv):
        deriv_conv1 = deriv['deriv_conv1']
        deriv_conv2 = deriv['deriv_conv2']
        deriv_full1 = deriv['deriv_full1']
        deriv_full2 = deriv['deriv_full2']

        self.conv_1_weights = self.conv_1_weights - self.learning_rate * deriv_conv1
        self.conv_2_weights = self.conv_2_weights - self.learning_rate * deriv_conv2
        self.full_1_weights = self.full_1_weights - self.learning_rate * deriv_full1
        self.full_2_weights = self.full_2_weights - self.learning_rate * deriv_full2

    def update_parameters(self, derivatives):
        deriv_conv1 = derivatives['deriv_conv1']
        deriv_conv2 = derivatives['deriv_conv2']
        deriv_full1 = derivatives['deriv_full1']
        deriv_full2 = derivatives['deriv_full2']

        self.conv_1_weights = self.conv_1_weights - self.learning_rate * deriv_conv1
        self.conv_2_weights = self.conv_2_weights - self.learning_rate * deriv_conv2
        self.full_1_weights = self.full_1_weights - self.learning_rate * deriv_full1
        self.full_2_weights = self.full_2_weights - self.learning_rate * deriv_full2

    '''
    Description:
        Cost function used here is the Categorical Cross Entropy function
        returns the cost for each image

    Parameters:
        labels: [batch_size][num_labels] The correct labels
        output: [batch_size][num_labes] the prediction from forward step

    Returns:
        loss: []
        accuracy:
    '''
    def calculate_cost(self, labels, output):
        loss = np.ndarray(labels.shape)
        for n in range(len(labels)):
            loss[n] = -np.sum(labels[n] * np.log(output))

        # n = len(labels)
        # loss1 = np.sum(np.multiply(np.log(output), labels), 1)
        # loss2 = np.sum(np.multiply(np.log(1 - output), 1 - labels), 1)
        # loss = np.sum((loss1 + loss2) * -1)
        # loss = loss + self.regularization * (np.sum(self.conv_1_weights**2)\
        #         + np.sum(self.conv_2_weights**2) + np.sum(self.full_1_weights**2)\
        #         + np.sum(self.full_2_weights**2))
        # loss = loss / n

        accuracy = np.sum(np.argmax(labels, 1)==np.argmax(output, 1)) / n

        return loss, accuracy

#
#         UTILITY FUNCTIONS
#
    def relu_layer(self, arr):
        arr[arr<=0] = 0

    def fully_conn_layer(self, arr, w, b):
        return np.matmul(arr, w) + b


    def expand(self, arr, size):
        return np.array(arr).reshape(size,(self.image_size // 4 - 1) * \
                    (self.image_size // 4 - 1) * self.depth * 4)


