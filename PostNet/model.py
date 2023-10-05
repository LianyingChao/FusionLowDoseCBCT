import tensorflow as tf
from utils import *


def generator(input):

    ##########################===============Encoder====================################################
    #layer0
    output1 = tf.layers.conv3d(input, 32, 3, padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer(), name="G_conv1")
    output1 = tf.nn.leaky_relu(output1)  
    output1 = tf.layers.conv3d(output1, 32, 3, padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer(), name="G_conv2")
    output1 = tf.nn.leaky_relu(output1)
    cal1 = output1
    # layer0 downsampling 
    output1 = tf.layers.conv3d(output1, 64, 3, padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer(), name="G_conv3")
    output1 = tf.nn.leaky_relu(output1)
    # output1=tf.layers.max_pooling3d(output1,3,(2,2,1),padding='SAME',name="G_maxpool1")
    output1 = tf.layers.conv3d(output1, 64, 3, (2,2,1),padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer(), name="G_downconv3")



    ###layer1
    output1 = tf.layers.conv3d(output1, 64, 3, padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer(), name="G_conv4")
    output1 = tf.nn.leaky_relu(output1)  
    output1 = tf.layers.conv3d(output1, 64, 3, padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer(), name="G_conv5")
    output1 = tf.nn.leaky_relu(output1)
    cal2 = output1
    # layer1 downsampling 
    output1 = tf.layers.conv3d(output1, 128, 3, padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer(), name="G_conv6")
    output1 = tf.nn.leaky_relu(output1)
    # output1=tf.layers.max_pooling3d(output1,3,(2,2,1),padding='SAME',name="G_maxpool2")
    output1 = tf.layers.conv3d(output1, 128, 3, (2,2,1),padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer(), name="G_downconv6")



    ###layer2
    output1 = tf.layers.conv3d(output1, 128, 3, padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer(), name="G_conv7")
    output1 = tf.nn.leaky_relu(output1)  
    output1 = tf.layers.conv3d(output1, 128, 3, padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer(), name="G_conv8")
    output1 = tf.nn.leaky_relu(output1)



    ##########################===============Decoder====================################################
    ###layer3
    output1 = tf.layers.conv3d(output1, 128, 3, padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer(), name="G_conv9")
    output1 = tf.nn.leaky_relu(output1)  
    output1 = tf.layers.conv3d(output1, 128, 3, padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer(), name="G_conv10")
    output1 = tf.nn.leaky_relu(output1)



    # layer4 upsampling 
    output1 = tf.layers.conv3d_transpose(output1, 128, 3,strides=(2,2,1), padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer(), name="G_deconv1")
    output1 = tf.nn.leaky_relu(output1)
    output1 = tf.layers.conv3d(output1, 64, 3, padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer(), name="G_conv11")
    ###layer4
    output1 = tf.layers.conv3d(output1, 64, 3, padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer(), name="G_conv12")
    output1 = tf.nn.leaky_relu(output1)  
    output1 = tf.concat([cal2,output1],axis=-1)
    output1 = tf.layers.conv3d(output1, 64, 3, padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer(), name="G_conv13")
    output1 = tf.nn.leaky_relu(output1)
    

    # layer5 upsampling 
    output1 = tf.layers.conv3d_transpose(output1, 64, 3,strides=(2,2,1), padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer(), name="G_deconv2")
    output1 = tf.nn.leaky_relu(output1)
    output1 = tf.layers.conv3d(output1, 32, 3, padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer(), name="G_conv14")
    ###layer5
    output1 = tf.layers.conv3d(output1, 32, 3, padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer(), name="G_conv15")
    output1 = tf.nn.leaky_relu(output1)  
    output1 = tf.concat([cal1,output1],axis=-1)
    output1 = tf.layers.conv3d(output1, 32, 3, padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer(), name="G_conv16")
    output1 = tf.nn.leaky_relu(output1)

    
    # output layer
    output1 = tf.layers.conv3d(output1, 1, 3, padding='SAME', kernel_initializer=tf.contrib.layers.xavier_initializer(), name="G_conv17")
    output1 = tf.nn.tanh(output1)
    output=input-output1
    return output
