import os
import re
import sys
import glob
from module import *
import scipy.misc
from itertools import cycle
import numpy as np
import tensorflow as tf
import imageio
from random import shuffle
# from libs import vgg16
from vgg16 import *
from PIL import Image

# from libs import vgg16
from vgg16 import *
from PIL import Image

LEARNING_RATE = 0.0002
BATCH_SIZE = 1
BATCH_SHAPE = [BATCH_SIZE, 640, 640, 9]
SKIP_STEP = 20
N_EPOCHS = 200
N_IMAGES = 3288
CKPT_DIR = './Checkpoints/'
IMG_DIR = './Images/'
GRAPH_DIR = './Graphs/'
TRAINING_SET_DIR= './dataset/ld/'
GROUNDTRUTH_SET_DIR= './dataset/nd/'
VALIDATION_SET_DIR='../dataset/validation/'
METRICS_SET_DIR='./dataset/metrics/'
TEST_SET_DIR='./dataset/test/'
TEST_SET_DIR_RESULT='./ImageResults/'
TRAINING_DIR_LIST = []
ADVERSARIAL_LOSS_FACTOR = 0.5
VGG_LOSS_FACTOR=100
PSNR_LOSS_FACTOR = -1.0
SSIM_LOSS_FACTOR = -1
MSE_LOSS_FACTOR = 1
CLIP = [-0.01,0.01]
CRITIC_NUM = 1

def initialize(sess):
    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(GRAPH_DIR, sess.graph)
    if not os.path.exists(CKPT_DIR):
        os.makedirs(CKPT_DIR)
    if not os.path.exists(IMG_DIR):
        os.makedirs(IMG_DIR)
    ckpt = tf.train.get_checkpoint_state(os.path.dirname(CKPT_DIR))
    if ckpt and ckpt.model_checkpoint_path:
        saver.restore(sess, ckpt.model_checkpoint_path)
    return saver


def imsave(filename, image):
    imageio.imsave(IMG_DIR+filename+'.tif', image)

def tryint(s):
    try:
        return int(s)
    except:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def split(arr, size):
    arrs = []
    while len(arr) > size:
        pice = arr[:size]
        arrs.append(pice)
        arr = arr[size:]
    arrs.append(arr)
    return arrs

def lrelu(x, leak=0.2, name='lrelu'):
    with tf.variable_scope(name):
        f1 = 0.5 * (1 + leak)
        f2 = 0.5 * (1 - leak)
        return f1 * x + f2 * abs(x)

def tf_log10(x):
    numerator = tf.log(x)
    denominator = tf.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator

def MSE(y_true, y_pred):
    return tf.reduce_mean(tf.square(y_pred - y_true))

def PSNR(y_true, y_pred):
    max_pixel = 1
    return 10.0 * tf_log10((max_pixel ** 2) / (tf.reduce_mean(tf.square(y_pred - y_true))))

def _tf_fspecial_gauss(size, sigma=1.5):
    """Function to mimic the 'fspecial' gaussian MATLAB function"""
    x_data, y_data = np.mgrid[-size//2 + 1:size//2 + 1, -size//2 + 1:size//2 + 1]
    x_data = np.expand_dims(x_data, axis=-1)
    x_data = np.expand_dims(x_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)
    y_data = np.expand_dims(y_data, axis=-1)

    x = tf.constant(x_data, dtype=tf.float32)
    y = tf.constant(y_data, dtype=tf.float32)
    g = tf.exp(-((x**2 + y**2)/(2.0*sigma**2)))
    return g / tf.reduce_sum(g)

def VGG_LOSS(img1, img2):

    a=img1[:,:,:,0,:]
    b=img2[:,:,:,0,:]
    a=tf.concat([a]*3, axis=3)
    b=tf.concat([b]*3, axis=3)
    [w, h, d] = a.get_shape().as_list()[1:]
    vgg_loss=tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square((extract_feature(a) -  extract_feature(b))))) / (w*h*d))

    for i in range(8):
        a = img1[:,:,:,i+1,:]
        b = img2[:,:,:,i+1,:]
        a = tf.concat([a]*3, axis=3)
        b = tf.concat([b]*3, axis=3)
        [w, h, d] = a.get_shape().as_list()[1:]
        vgg_loss = vgg_loss + tf.reduce_mean(tf.sqrt(tf.reduce_sum(tf.square((extract_feature(a) -  extract_feature(b))))) / (w*h*d))


    return vgg_loss/9

def SSIM_one(img1, img2, k1=0.01, k2=0.02, L=1, window_size=11):
    """
    The function is to calculate the ssim score
    """
    # img1 = tf.expand_dims(img1, -1)
    # print(img1.shape)
    # img2 = tf.expand_dims(img2, -1)
    # print(img2.shape)
    window = _tf_fspecial_gauss(window_size)
    mu1 = tf.nn.conv2d(img1, window, strides = [1, 1, 1, 1], padding = 'VALID')
    mu2 = tf.nn.conv2d(img2, window, strides = [1, 1, 1, 1], padding = 'VALID')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2

    sigma1_sq = tf.nn.conv2d(img1*img1, window, strides = [1 ,1, 1, 1], padding = 'VALID') - mu1_sq
    sigma2_sq = tf.nn.conv2d(img2*img2, window, strides = [1, 1, 1, 1], padding = 'VALID') - mu2_sq
    sigma1_2 = tf.nn.conv2d(img1*img2, window, strides = [1, 1, 1, 1], padding = 'VALID') - mu1_mu2
    c1 = (k1*L)**2
    c2 = (k2*L)**2
    ssim_map = ((2*mu1_mu2 + c1)*(2*sigma1_2 + c2)) / ((mu1_sq + mu2_sq + c1)*(sigma1_sq + sigma2_sq + c2))
    return tf.reduce_mean(ssim_map)

def SSIM_three(img1, img2):
    rgb1 = tf.unstack(img1, axis=3)
    rgb2 = tf.unstack(img2, axis=3)
    ssim = SSIM_one(rgb1[0], rgb2[0])
    for i in range(8):
        ssim = ssim + SSIM_one(rgb1[i], rgb2[i])
    return ssim/9

path='/mnt/data2/fusion_wal/data/'
patient_folder = ["1","2","3","4","5","6","7","8","9","10","11", "12","13","14","15","16","17"]


def tr_in(idx):
    projs_nd = np.zeros((448, 448, 200), dtype=np.float32)
    path_read_nd=path+patient_folder[idx]+"/nd_re/"
    projs_ld = np.zeros((448, 448, 200), dtype=np.float32)
    path_read_ld=path+patient_folder[idx]+"/dldrsv112_re/"
    for i in range(200):
        #print(i)
        a=imageio.imread(path_read_nd + '%d.png'%(i+1)).astype(float)
        projs_nd[:,:,i]=(a-a.min())/(a.max()-a.min())
        b=imageio.imread(path_read_ld + '%d.png'%(i+1)).astype(float)
        projs_ld[:,:,i]=(b-b.min())/(b.max()-b.min())
    return projs_nd, projs_ld


