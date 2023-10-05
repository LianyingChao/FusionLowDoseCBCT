import time
import tensorflow as tf
import numpy as np
from utils import *
from model import *
from skimage import measure
import scipy.io as io
import imageio



def test():
    tf.reset_default_graph()
    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
    gen_in = tf.placeholder(shape=[1, 448, 448, 9, 1], dtype=tf.float32,
                            name='generated_image')
    Gz = generator(gen_in)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        saver = initialize(sess)
        initial_step = global_step.eval()
        interval = 0
        for cc in range(1):
            #idx=cc+19
            #print(idx)
            path = '../pre_CBCT/'
            path_save = '../post_CBCT/'
            if not os.path.exists(path_save):
                os.makedirs(path_save)

            recon = np.zeros((448, 448, 200), dtype=np.float32)
            for cc in range(200):
                print(cc)
                a=imageio.imread(path + '%d.png'%(cc+1)).astype(float)
                recon[:,:,cc] = (a-a.min())/(a.max()-a.min())

            print('Testing')
            batch = np.zeros((448, 448, 9, 1), dtype=np.float32)
            batch[:,:,0,0] = recon[:,:,0]
            batch[:,:,1,0] = recon[:,:,0]
            batch[:,:,2,0] = recon[:,:,0]
            batch[:,:,3,0] = recon[:,:,0]
            batch[:,:,4,0] = recon[:,:,0]
            batch[:,:,5,0] = recon[:,:,1]
            batch[:,:,6,0] = recon[:,:,2]
            batch[:,:,7,0] = recon[:,:,3]
            batch[:,:,8,0] = recon[:,:,4]
            image = np.expand_dims(batch,axis=0)
            image_recon = sess.run(Gz, feed_dict={gen_in: image})
            image_recon = np.resize(image_recon,[448,448,9])
            img = image_recon[:,:,4]
            img[img < 0.0]=0.0
            img[img > 1.0]=1.0
            imageio.imsave(path_save+'1.png', img)

            batch = np.zeros((448, 448, 9, 1), dtype=np.float32)
            batch[:,:,0,0] = recon[:,:,0]
            batch[:,:,1,0] = recon[:,:,0]
            batch[:,:,2,0] = recon[:,:,0]
            batch[:,:,3,0] = recon[:,:,0]
            batch[:,:,4,0] = recon[:,:,1]
            batch[:,:,5,0] = recon[:,:,2]
            batch[:,:,6,0] = recon[:,:,3]
            batch[:,:,7,0] = recon[:,:,4]
            batch[:,:,8,0] = recon[:,:,5]
            image = np.expand_dims(batch,axis=0)
            image_recon = sess.run(Gz, feed_dict={gen_in: image})
            image_recon = np.resize(image_recon,[448,448,9])
            img = image_recon[:,:,4]
            img[img < 0.0]=0.0
            img[img > 1.0]=1.0
            imageio.imsave(path_save+'2.png', img)


            batch = np.zeros((448, 448, 9, 1), dtype=np.float32)
            batch[:,:,0,0] = recon[:,:,0]
            batch[:,:,1,0] = recon[:,:,0]
            batch[:,:,2,0] = recon[:,:,0]
            batch[:,:,3,0] = recon[:,:,1]
            batch[:,:,4,0] = recon[:,:,2]
            batch[:,:,5,0] = recon[:,:,3]
            batch[:,:,6,0] = recon[:,:,4]
            batch[:,:,7,0] = recon[:,:,5]
            batch[:,:,8,0] = recon[:,:,6]
            image = np.expand_dims(batch,axis=0)
            image_recon = sess.run(Gz, feed_dict={gen_in: image})
            image_recon = np.resize(image_recon,[448,448,9])
            img = image_recon[:,:,4]
            img[img < 0.0]=0.0
            img[img > 1.0]=1.0
            imageio.imsave(path_save+'3.png', img)



            batch = np.zeros((448, 448, 9, 1), dtype=np.float32)
            batch[:,:,0,0] = recon[:,:,0]
            batch[:,:,1,0] = recon[:,:,0]
            batch[:,:,2,0] = recon[:,:,1]
            batch[:,:,3,0] = recon[:,:,2]
            batch[:,:,4,0] = recon[:,:,3]
            batch[:,:,5,0] = recon[:,:,4]
            batch[:,:,6,0] = recon[:,:,5]
            batch[:,:,7,0] = recon[:,:,6]
            batch[:,:,8,0] = recon[:,:,7]
            image = np.expand_dims(batch,axis=0)
            image_recon = sess.run(Gz, feed_dict={gen_in: image})
            image_recon = np.resize(image_recon,[448,448,9])
            img = image_recon[:,:,4]
            img[img < 0.0]=0.0
            img[img > 1.0]=1.0
            imageio.imsave(path_save+'4.png', img)


            batch = np.zeros((448, 448, 9, 1), dtype=np.float32)
            for i in range(192):
                print(i)
                batch[:,:,0,0] = recon[:,:,i]
                batch[:,:,1,0] = recon[:,:,i+1]
                batch[:,:,2,0] = recon[:,:,i+2]
                batch[:,:,3,0] = recon[:,:,i+3]
                batch[:,:,4,0] = recon[:,:,i+4]
                batch[:,:,5,0] = recon[:,:,i+5]
                batch[:,:,6,0] = recon[:,:,i+6]
                batch[:,:,7,0] = recon[:,:,i+7]
                batch[:,:,8,0] = recon[:,:,i+8]
                image = np.expand_dims(batch,axis=0)
                image_recon = sess.run(Gz, feed_dict={gen_in: image})
                image_recon = np.resize(image_recon,[448,448,9])
                img = image_recon[:,:,4]
                img[img < 0.0]=0.0
                img[img > 1.0]=1.0
                imageio.imsave(path_save+'/%d.png'%(i+5), img)



            batch = np.zeros((448, 448, 9, 1), dtype=np.float32)
            batch[:,:,0,0] = recon[:,:,192]
            batch[:,:,1,0] = recon[:,:,193]
            batch[:,:,2,0] = recon[:,:,194]
            batch[:,:,3,0] = recon[:,:,195]
            batch[:,:,4,0] = recon[:,:,196]
            batch[:,:,5,0] = recon[:,:,197]
            batch[:,:,6,0] = recon[:,:,198]
            batch[:,:,7,0] = recon[:,:,199]
            batch[:,:,8,0] = recon[:,:,199]
            image = np.expand_dims(batch,axis=0)
            image_recon = sess.run(Gz, feed_dict={gen_in: image})
            image_recon = np.resize(image_recon,[448,448,9])
            img = image_recon[:,:,4]
            img[img < 0.0]=0.0
            img[img > 1.0]=1.0
            imageio.imsave(path_save+'197.png', img)


            batch = np.zeros((448, 448, 9, 1), dtype=np.float32)
            batch[:,:,0,0] = recon[:,:,193]
            batch[:,:,1,0] = recon[:,:,194]
            batch[:,:,2,0] = recon[:,:,195]
            batch[:,:,3,0] = recon[:,:,196]
            batch[:,:,4,0] = recon[:,:,197]
            batch[:,:,5,0] = recon[:,:,198]
            batch[:,:,6,0] = recon[:,:,199]
            batch[:,:,7,0] = recon[:,:,199]
            batch[:,:,8,0] = recon[:,:,199]
            image = np.expand_dims(batch,axis=0)
            image_recon = sess.run(Gz, feed_dict={gen_in: image})
            image_recon = np.resize(image_recon,[448,448,9])
            img = image_recon[:,:,4]
            img[img < 0.0]=0.0
            img[img > 1.0]=1.0
            imageio.imsave(path_save+'198.png', img)



            batch = np.zeros((448, 448, 9, 1), dtype=np.float32)
            batch[:,:,0,0] = recon[:,:,194]
            batch[:,:,1,0] = recon[:,:,195]
            batch[:,:,2,0] = recon[:,:,196]
            batch[:,:,3,0] = recon[:,:,197]
            batch[:,:,4,0] = recon[:,:,198]
            batch[:,:,5,0] = recon[:,:,199]
            batch[:,:,6,0] = recon[:,:,199]
            batch[:,:,7,0] = recon[:,:,199]
            batch[:,:,8,0] = recon[:,:,199]
            image = np.expand_dims(batch,axis=0)
            image_recon = sess.run(Gz, feed_dict={gen_in: image})
            image_recon = np.resize(image_recon,[448,448,9])
            img = image_recon[:,:,4]
            img[img < 0.0]=0.0
            img[img > 1.0]=1.0
            imageio.imsave(path_save+'199.png', img)



            batch = np.zeros((448, 448, 9, 1), dtype=np.float32)
            batch[:,:,0,0] = recon[:,:,195]
            batch[:,:,1,0] = recon[:,:,196]
            batch[:,:,2,0] = recon[:,:,197]
            batch[:,:,3,0] = recon[:,:,198]
            batch[:,:,4,0] = recon[:,:,199]
            batch[:,:,5,0] = recon[:,:,199]
            batch[:,:,6,0] = recon[:,:,199]
            batch[:,:,7,0] = recon[:,:,199]
            batch[:,:,8,0] = recon[:,:,199]
            image = np.expand_dims(batch,axis=0)
            image_recon = sess.run(Gz, feed_dict={gen_in: image})
            image_recon = np.resize(image_recon,[448,448,9])
            img = image_recon[:,:,4]
            img[img < 0.0]=0.0
            img[img > 1.0]=1.0
            imageio.imsave(path_save+'200.png', img)

    
if __name__ == '__main__':
    test()
