import time
import tensorflow as tf
import numpy as np
from utils import *
from model import *
from skimage import measure
import scipy.io as io
import imageio
import matplotlib.pyplot as plt


def train():
    tf.reset_default_graph()
    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
    gen_in = tf.placeholder(shape=[None, 448, 448, 9,1], dtype=tf.float32,
                            name='generated_image')
    real_in = tf.placeholder(shape=[None, 448, 448, 9,1], dtype=tf.float32,
                             name='groundtruth_image')
    ###========================== DEFINE MODEL ============================###
    Gz = generator(gen_in)
    # G_loss = VGG_LOSS_FACTOR * VGG_LOSS(real_in, Gz) + SSIM_LOSS_FACTOR*SSIM_three(real_in, Gz)
    G_loss = MSE_LOSS_FACTOR* MSE(real_in, Gz)
    t_vars = tf.trainable_variables()
    G_vars = [var for var in t_vars if 'G_' in var.name]
    G_solver = tf.train.AdamOptimizer(LEARNING_RATE, beta1 = 0.5, beta2 = 0.9).minimize(G_loss, var_list=G_vars)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)
        saver = initialize(sess)
        initial_step = global_step.eval()
        step = 1
        v=0
        for epoch in range(N_EPOCHS):
            patient = [int(x) for x in list(range(0,17,1))]
            # print(idx)
            # quit()
            shuffle(patient)
            idx=patient[15]
            # for idx in patient:    
            Refen, Input=tr_in(idx)
            # print(Refen.shape)
            # print(Input.shape)
            # print(idx)
            # quit()
            sli = [int(y) for y in list(range(0,190,1))]
            shuffle(sli)
            training_batch = np.zeros((1, 448, 448, 9, 1), dtype=np.float32)
            groundtruth_batch = np.zeros((1, 448, 448, 9, 1), dtype=np.float32)
            s=0
            for i in sli:
                s=s+1
                print(s)
                print(i)
                print(idx)
                print(10*'*')
                training_batch[0,:,:,:,0] = Input[:,:,i:i+9]
                groundtruth_batch[0,:,:,:,0] = Refen[:,:,i:i+9]
                # for k in range(9):
                #     print(training_batch[0,:,:,k,0].dtype,groundtruth_batch[0,:,:,k,0].dtype)
                #     print(training_batch[0,:,:,k,0].max(),training_batch[0,:,:,k,0].min())
                #     print(groundtruth_batch[0,:,:,k,0].max(),groundtruth_batch[0,:,:,k,0].min())
                #     # imageio.imsave('%d-ld.png'%(k+1), training_batch[0,:,:,k,0])
                #     imageio.imsave('%d-nd.png'%(k+1), groundtruth_batch[0,:,:,k,0])
                # quit()
                step += 1
                _, G_loss_cur = sess.run([G_solver, G_loss], feed_dict={gen_in: training_batch, real_in: groundtruth_batch})
                print("epoch: {} ".format(epoch + 1) + "pro_id:{} Gen Loss: ".format(i) + str(G_loss_cur))
                
                if step % 600 == 0:
                    #v=v+1
                    saver.save(sess, CKPT_DIR, step)
                    '''
                    os.makedirs('./wal18/%d/proj'%v)
                    os.makedirs('./wal18/%d/re'%v)

                    #validation projections
                    for m in range(500):
                        print(m)
                        image = np.expand_dims(np.load('./wal18/test/%d.npy'%(m+1)),axis=0)
                        image_recon = sess.run(Gz, feed_dict={gen_in: image})
                        image_recon=np.resize(image_recon,[544,576,9])
                        imageio.imsave('./wal18/%d/proj/'%v +'%d.tif'%(m+1), image_recon[:,:,4])
                    '''


if __name__ == '__main__':
    train()
