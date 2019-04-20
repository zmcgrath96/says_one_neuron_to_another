import numpy as np
from scipy.special import softmax
import sys
import time


def predict(img, params):
    pool_width = 2
    pool_stride = 2
    conv_stride = 1
    [filter_1, filter_2, weight_1, weight_2, bias_1, bias_2, bias_3, bias_4] = params

    conv1 = forward_conv(img, filter_1, bias_1, conv_stride)
    conv1[conv1<=0] = 0 # RELU
    conv2 = forward_conv(conv1, filter_2, bias_2, conv_stride)
    conv2[conv2<=0] = 0 # RELU

    pooled = forward_maxpool(conv2, pool_width, pool_stride)
    (y, x, d) = pooled.shape
    fully_conn = pooled.reshape((d * x * y, 1))

    # condensing the fully connected layer
    dense_1 = weight_1.dot(fully_conn) + bias_3
    dense_1[dense_1<=0] = 0

    output = weight_2.dot(dense_1) + bias_4

    return softmax(output)

'''
Description:
    Train the CNN over the given dataset

Parameters:
    data: dataset to be trained over
    labels: label for each data point. Assumed to have same index
    num_classes: number of classes in output
    img_dim: length and width of image
    img_depth: number of colors in image
    batch_size: number of images to process before back propagation
    epochs: number of training iterations
'''
def train(data, labels, num_classes, img_dim=250, img_depth=3, batch_size=20, epochs=1):
    filt_dim = 5
    num_filt = 2
    pool_width = 2
    pool_stride = 2
    conv_stride = 1
    a = int((img_dim - filt_dim)/conv_stride) + 1
    x = int((a - filt_dim)/conv_stride) + 1
    weight_1_size = int((x - pool_width)/pool_stride) + 1

    filter_1 = initialize_filter((num_filt,filt_dim,filt_dim,img_depth))
    filter_2 = initialize_filter((num_filt,filt_dim,filt_dim,num_filt))
    weight_1 = initialize_weights((128, num_filt * weight_1_size * weight_1_size))
    weight_2 = initialize_weights((num_classes,128))

    bias_1 = np.zeros((filter_1.shape[0],1))
    bias_2 = np.zeros((filter_2.shape[0],1))
    bias_3 = np.zeros((weight_1.shape[0],1))
    bias_4 = np.zeros((weight_2.shape[0],1))

    params = [filter_1, filter_2, weight_1, weight_2, bias_1, bias_2, bias_3, bias_4]

    cost = []

    for epoch in range(epochs):
         batches = [data[k:k + batch_size] for k in range(0, data.shape[0], batch_size)]
         i = 1
         for batch in batches:
             s_t = time.time()
             print("Epoch: {}, Batch: {}".format(epoch, i), end="")
             sys.stdout.flush()
             params, cost = adam_opt_alg(batch, labels, num_classes, 0.01, img_dim, img_depth, 0.9, 0.999, params, cost, conv_stride, pool_width, pool_stride)
             print("\n {}".format(cost))
             sys.stdout.flush()
             end_t = time.time()
             print("Batch {} took {} s".format(i, end_t - s_t))
             i += 1
    return params

'''
Description:
    Initialize a filter of given size array

Parameters:
    size: a tuple of integers for the size of the filter

Returns:
    an numpy array of the given dimensions with a normal distribution
'''
def initialize_filter(size):
    std_dev = 1 / np.sqrt(np.prod(size))
    return np.random.normal(loc=0, scale=std_dev, size=size)

'''
Description:
    Initializes a numpy array of the given size

Parameters:
    size: a tuple of the dimensions of the desired array

Returns:
    a numpy array initialized with a notmal distribution
'''
def initialize_weights(size):
    return np.random.standard_normal(size=size) * 0.01

'''
Description:
     Adam optimization algorithm

Parameters:
    batch: array of images currently being batched
    labels: label for each data point. Assumed to have same index
    num_classes: number of classes in output
    lr: learning rate
    img_dim: image width, height
    d: image depth, number of colors
    beta1: exponential decay rate for the first moment estimates
    beta2: exponential decay rate for the second-moment estimates
    params: filters and weights
    cost: list of costs

Returns:
    params: array of the filters, weights, and biases after back propogation
    cost: total cost for the batch
'''
def adam_opt_alg(batch, labels, num_classes, lr, img_dim, d, beta1, beta2, params, cost, conv_stride, pool_width, pool_stride):
    [filter_1, filter_2, weight_1, weight_2, bias_1, bias_2, bias_3, bias_4] = params
    epsilon = 1e-7

    temp_cost = 0
    batch_size = len(batch)

    # initialize gradients and momentum,RMS params
    deriv_filter_1 = np.zeros(filter_1.shape)
    deriv_filter_2 = np.zeros(filter_2.shape)
    deriv_weight_1 = np.zeros(weight_1.shape)
    deriv_weight_2 = np.zeros(weight_2.shape)
    deriv_bias_1 = np.zeros(bias_1.shape)
    deriv_bias_2 = np.zeros(bias_2.shape)
    deriv_bias_3 = np.zeros(bias_3.shape)
    deriv_bias_4 = np.zeros(bias_4.shape)

    v1 = np.zeros(filter_1.shape)
    v2 = np.zeros(filter_2.shape)
    v3 = np.zeros(weight_1.shape)
    v4 = np.zeros(weight_2.shape)
    bv1 = np.zeros(bias_1.shape)
    bv2 = np.zeros(bias_2.shape)
    bv3 = np.zeros(bias_3.shape)
    bv4 = np.zeros(bias_4.shape)

    s1 = np.zeros(filter_1.shape)
    s2 = np.zeros(filter_2.shape)
    s3 = np.zeros(weight_1.shape)
    s4 = np.zeros(weight_2.shape)
    bs1 = np.zeros(bias_1.shape)
    bs2 = np.zeros(bias_2.shape)
    bs3 = np.zeros(bias_3.shape)
    bs4 = np.zeros(bias_4.shape)

    for i in range(batch_size):
        print(".", end="")
        sys.stdout.flush()

        img = batch[i]
        label = labels[i]

        grads, loss = convolution(img, label, params, conv_stride, pool_width, pool_stride)
        [temp_deriv_filter_1, temp_deriv_filter_2, temp_deriv_weight_1, temp_deriv_weight_2, temp_deriv_bias_1, temp_deriv_bias_2, temp_deriv_bias_3, temp_deriv_bias_4] = grads

        deriv_filter_1 += temp_deriv_filter_1
        deriv_bias_1 += temp_deriv_bias_1
        deriv_filter_2 += temp_deriv_filter_2
        deriv_bias_2 += temp_deriv_bias_2
        deriv_weight_1 += temp_deriv_weight_1
        deriv_bias_3 += temp_deriv_bias_3
        deriv_weight_2 += temp_deriv_weight_2
        deriv_bias_4 += temp_deriv_bias_4

        temp_cost += loss

    v1 = beta1 * v1 + (1-beta1) * deriv_filter_1 / batch_size
    s1 = beta2 * s1 + (1-beta2) * (deriv_filter_1 / batch_size)**2
    filter_1 -= lr * v1/np.sqrt(s1+epsilon)

    bv1 = beta1 * bv1 + (1 - beta1) * deriv_bias_1 / batch_size
    bs1 = beta2 * bs1 + (1 - beta2) * (deriv_bias_1 / batch_size)**2
    bias_1 -= lr * bv1/np.sqrt(bs1 + epsilon)

    v2 = beta1 * v2 + (1 - beta1) * deriv_filter_2 / batch_size
    s2 = beta2 * s2 + (1 - beta2) * (deriv_filter_2 / batch_size)**2
    filter_2 -= lr * v2 / np.sqrt(s2 + epsilon)

    bv2 = beta1 * bv2 + (1 - beta1) * deriv_bias_2 / batch_size
    bs2 = beta2 * bs2 + (1 - beta2) * (deriv_bias_2 / batch_size)**2
    bias_2 -= lr * bv2 / np.sqrt(bs2 + epsilon)

    v3 = beta1 * v3 + (1 - beta1) * deriv_weight_1 / batch_size
    s3 = beta2 * s3 + (1 - beta2) * (deriv_weight_1 / batch_size)**2
    weight_1 -= lr * v3 / np.sqrt(s3 + epsilon)

    bv3 = beta1 * bv3 + (1 - beta1) * deriv_bias_3 / batch_size
    bs3 = beta2 * bs3 + (1 - beta2) * (deriv_bias_3 / batch_size)**2
    bias_3 -= lr * bv3 / np.sqrt(bs3 + epsilon)

    v4 = beta1 * v4 + (1 - beta1) * deriv_weight_2 / batch_size
    s4 = beta2 * s4 + (1 - beta2) * (deriv_weight_2 / batch_size)**2
    weight_2 -= lr * v4 / np.sqrt(s4 + epsilon)

    bv4 = beta1 * bv4 + (1 - beta1)  *deriv_bias_4 / batch_size
    bs4 = beta2 * bs4 + (1 - beta2) * (deriv_bias_4 / batch_size)**2
    bias_4 -= lr * bv4 / np.sqrt(bs4 + epsilon)


    temp_cost = temp_cost/batch_size
    cost.append(temp_cost)

    params = [filter_1, filter_2, weight_1, weight_2, bias_1, bias_2, bias_3, bias_4]

    return params, cost

'''
Description:
    Does forward and back forward passes of CNN

Parameters:
    img: the image to be processed
    label: the one-hot array for the associated with image
    params: array of filters, weights, and biases
    conv_stride: stride of the conv layer filter
    pool_width: width of the pool layer
    pool_stride: stride of the pooling layer

Returns:
    grad: gradients for the img
    loss: loss over the image
'''
def convolution(img, label, params, conv_stride, pool_width, pool_stride):
    [filter_1, filter_2, weight_1, weight_2, bias_1, bias_2, bias_3, bias_4] = params

    '''
    Forward Operation
    '''

    conv1 = forward_conv(img, filter_1, bias_1, conv_stride)
    conv1[conv1<=0] = 0 # RELU
    conv2 = forward_conv(conv1, filter_2, bias_2, conv_stride)
    conv2[conv2<=0] = 0 # RELU

    pooled = forward_maxpool(conv2, pool_width, pool_stride)
    (y, x, d) = pooled.shape
    fully_conn = pooled.reshape((d * x * y, 1))

    # condensing the fully connected layer
    dense_1 = weight_1.dot(fully_conn) + bias_3
    dense_1[dense_1<=0] = 0

    output = weight_2.dot(dense_1) + bias_4

    probs = softmax(output)


    '''
    Compute loss
    '''
    loss = calc_loss(probs, label)

    '''
    Backward propogation
    '''
    delta_out = probs - label.reshape(probs.shape)
    deriv_weight_2 = delta_out.dot(dense_1.T)
    deriv_bias_4 = np.sum(delta_out, axis = 1).reshape(bias_4.shape)

    deriv_dense_1 = weight_2.T.dot(delta_out)
    deriv_dense_1[dense_1<=0] = 0

    deriv_weight_1 = deriv_dense_1.dot(fully_conn.T)
    deriv_bias_3 = np.sum(deriv_dense_1, axis=1).reshape(bias_3.shape)

    deriv_fully_conn = weight_1.T.dot(deriv_dense_1)
    deriv_pool = deriv_fully_conn.reshape(pooled.shape)

    deriv_conv2 = backward_maxpool(deriv_pool, conv2, pool_width, pool_stride)
    deriv_conv2[conv2 <= 0] = 0

    deriv_conv1, deriv_filter_2, deriv_bias_2 = backward_conv(deriv_conv2, conv1, filter_2, conv_stride)
    deriv_conv1[conv1 <= 0] = 0

    deriv_img, deriv_filter_1, deriv_bias_1 = backward_conv(deriv_conv1, img, filter_1, conv_stride)

    gradients = [deriv_filter_1, deriv_filter_2, deriv_weight_1, deriv_weight_2, deriv_bias_1, deriv_bias_2, deriv_bias_3, deriv_bias_4]

    return gradients, loss

'''
Description:
    The loss function. Here we use the categorical cross entropy function

Parameters:
    output [num_classes]: the predicted output from the CNN (probability distribution)
    labels [num_classes]: one hot array that has the actual value of the image

Returns:
    loss int
'''
def calc_loss(output, labels):
    return -np.sum(labels * np.log(output.clip(min=0.00000001)))

'''
=================================================================================================
                            FORWARD STUFF
=================================================================================================
'''

'''
Description:
    Convolutes an image. This does the forward steps. The bias and filter values
    are updated in the reverse step

Parameters:
    img: the numpy array of [y, x, z] or [height, width, rgb]
    filter: the filters used in convolution [num_filter, filter_width, filter_height, num_color]
    bias: bias value for each filter, [num_filter]
    stride: int for how far to move the filter across the image

Returns:
    convoluted image: numpy array [(y - f_y)/stride + 1, (x-f_x)/stride + 1,num_filters]
'''
def forward_conv(img, filter, bias, stride = 1):
    (y, x, d) = img.shape
    (n_f, f_y, f_x, f_d) = filter.shape
    convoluted_dims = int((y - f_x)/stride) + 1
    convoluted = np.zeros((convoluted_dims, convoluted_dims, n_f))

    # go through each filter
    for f in range(n_f):
        # move vertically about image
        curr_y = conv_y = 0
        while curr_y + f_y <= y:
            # move horizontally about image
            curr_x = conv_x = 0
            while curr_x + f_x <= x:
                convoluted[conv_y, conv_x, f] = np.sum(filter[f] * img[curr_y:curr_y+f_y, curr_x:curr_x+f_x,:]) + bias[f]
                curr_x += stride
                conv_x += 1
            curr_y += stride
            conv_y += 1

    return convoluted

'''
Description:
    Down samples the input np array, or the convoluted image for feature extraction

Parameters:
    convoluted [y, x, d]: The convoluted image, np array, to down sample
    pool_width int: the size of the pooling window in which to extract the max
    stride int: how far to move the pooling window for each step

Returns:
    the pooled, downsampled image
    [downsampled y, downsampled x, depth]
'''
def forward_maxpool(convoluted, pool_width, stride):
    (y, x, d) = convoluted.shape
    new_y = int((y - pool_width)/stride) + 1
    new_x = int((x - pool_width)/stride) + 1

    pooled = np.zeros((new_y, new_x, d))

    # go through the depth of the convoluted
    for i in range(d):
        # go through the height of the convoluted
        curr_y = p_y = 0
        while curr_y + pool_width <= y:
            # go through the width of the convoluted
            curr_x = p_x = 0
            while curr_x + pool_width <= x:
                # get the max of the window over the convoluted image
                pooled[p_y, p_x, i] = np.max(convoluted[curr_y:curr_y + pool_width, curr_x:curr_x + pool_width, i])

                curr_x += stride
                p_x += 1
            curr_y += stride
            p_y += 1

    return pooled

'''
=================================================================================================
                            BACKWARD STUFF
=================================================================================================
'''

'''
Description:
    Calculates the loss gradient of the filter, the convolution, and biases
    Essentially how wrong all of our weights and biases are

Parameters:
    derivs_conv_prev: loss gradient of the previous convolution
    conv: the current convolution to calculated loss gradient
    filter: the filters used here to calculate loss gradient
    stride: how far across the input convolution to move

Returns:
    derivs_out [c_y, c_x, c_d]: loss gradient of the input conv
    derivs_f [n_f, f_y, f_x, f_d]: loss gradient of the filter
    derivs_b [n_f]: loss gradient of the biases

'''
def backward_conv(derivs_conv_prev, conv, filter, stride=1):
    (n_f, f_y, f_x, f_d) = filter.shape
    (c_y, c_x, c_d) = conv.shape

    derivs_out = np.zeros(conv.shape)
    derivs_f = np.zeros(filter.shape)
    derivs_b = np.zeros((n_f, 1))

    # go through each filter
    for f in range(n_f):
        # go through the convoluted vertically
        curr_y = d_y = 0
        while curr_y + f_y <= c_y:
            # go through the convoluted horizontally
            curr_x = d_x = 0
            while curr_x + f_x <= c_x:
                derivs_f[f] = derivs_conv_prev[d_y, d_x, f] * conv[curr_y:curr_y + f_y, curr_x:curr_x + f_x, :]
                derivs_out[curr_y:curr_y + f_y, curr_x:curr_x + f_x, :] = derivs_conv_prev[d_y, d_x, f] * filter[f]

                curr_x += stride
                d_x += 1
            curr_y += stride
            d_y += 1

        derivs_b[f] = np.sum(derivs_conv_prev[f])

    return derivs_out, derivs_f, derivs_b

'''

'''
# def arg

'''
Description:
    Calculates the loss / impact on error the pooling had

Parameters:
    derivs_pool: derivatives/loss gradient of the pooled convolution in the forward step
    conv: the convoluted image that the maxpool worked on forward
    pool_width: width of the pooling window
    stride: how far to step the window

Returns:
    derivs_out: the loss gradient of the pooled layer
'''
def backward_maxpool(derivs_pool, conv, pool_width, stride):
    (c_y, c_x, c_d) = conv.shape

    derivs_out = np.zeros(conv.shape)

    # go through the depth of the convoluted layer
    for d in range(c_d):
        # go through the height of the convoluted
        curr_y = d_y = 0
        for y in range(c_y):
            # go through the width of the convoluted
            curr_x = d_x = 0
            for x in range(c_x):
                # get the index in the convoluted where the max was located
                t = conv[curr_y:curr_y + pool_width, curr_x:curr_x + pool_width, d]
                if t.any():
                    (conv_y, conv_x) = np.unravel_index(np.nanargmax(t), t.shape)

                    derivs_out[curr_y + conv_y, curr_x + conv_x, d] = derivs_pool[d_y, d_x, d]

                curr_x += stride
                d_x += 1
            curr_y += stride
            d_y += 1

    return derivs_out





