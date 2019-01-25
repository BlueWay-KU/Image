# -*- coding: utf-8 -*-
import cv2
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from PIL import Image

import models

def dark_channel(img, size = 15):
    r, g, b = cv2.split(img)
    min_img = cv2.min(r, cv2.min(g, b))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    dc_img = cv2.erode(min_img,kernel)
    return dc_img

def get_atmo(img, percent = 0.001):
    mean_perpix = np.mean(img, axis = 2).reshape(-1)
    mean_topper = mean_perpix[:int(img.shape[0] * img.shape[1] * percent)]
    return np.mean(mean_topper)

def get_trans(img, atom, w = 0.95):
    x = img / atom
    t = 1 - w * dark_channel(x, 15)
    return t

def guided_filter(p, i, r, e):
    """
    p: input image
    i: guidance image
    r: radius
    e: regularization
    return q: filtering output
    """
    #1
    mean_I = cv2.boxFilter(i, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    corr_I = cv2.boxFilter(i * i, cv2.CV_64F, (r, r))
    corr_Ip = cv2.boxFilter(i * p, cv2.CV_64F, (r, r))
    #2
    var_I = corr_I - mean_I * mean_I
    cov_Ip = corr_Ip - mean_I * mean_p
    #3
    a = cov_Ip / (var_I + e)
    b = mean_p - a * mean_I
    #4
    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))
    #5
    q = mean_a * i + mean_b
    return q

def dehaze(path, output = None):
    im = cv2.imread(path)
    img = im.astype('float64') / 255
    img_gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype('float64') / 255

    atom = get_atmo(img)
    trans = get_trans(img, atom)
    trans_guided = guided_filter(trans, img_gray, 20, 0.0001)
    trans_guided = cv2.max(trans_guided, 0.25)

    result = np.empty_like(img)
    for i in range(3):
        result[:, :, i] = (img[:, :, i] - atom) / trans_guided + atom
    cv2.imwrite('output_dehaze.jpg', result * 255)

def predict(model_data_path, image_path):
    # Default input size
    height = 228
    width = 304
    channels = 3
    batch_size = 1

    # Read image
    img = Image.open(image_path)
    img = img.resize([width, height], Image.ANTIALIAS)
    img = np.array(img).astype('float32')
    img = np.expand_dims(np.asarray(img), axis=0)

    # Create a placeholder for the input image
    input_node = tf.placeholder(tf.float32, shape=(None, height, width, channels))

    # Construct the network
    net = models.ResNet50UpProj({'data': input_node}, batch_size, 1, False)

    with tf.Session() as sess:
        # Load the converted parameters
        # Use to load from ckpt file
        saver = tf.train.Saver()
        saver.restore(sess, model_data_path)

        # Use to load from npy file
        # net.load(model_data_path, sess)

        # Evalute the network for the given image
        pred = sess.run(net.get_output(), feed_dict={input_node: img})

        # Plot result
        #fig = plt.figure()
        #ii = plt.imshow(pred[0, :, :, 0], interpolation='nearest')
        #fig.colorbar(ii)
        #plt.show()

        return pred

result = dehaze('cones.jpg')
result = predict('NYU_FCRN.ckpt', 'output_dehaze.jpg')

x = len(result[0][0]) #160
y = len(result[0]) #128

# result = [0][127][159] = [0][y][x] m
#result[0][47:79][59:99]
x_a, x_m, x_b = x/2 - x/4, x/2, x/2 + x/4
y_a, y_m, y_b = y/2 - y/4, y/2, y/2 + y/4

for i in range(31, 96):
    for j in range(39, 80):
        cal = np.mean(result[0][i][j])

if cal < 2:
    print('Blocked')
elif cal >=2 :
    pass