import tensorflow as tf
import os
import sys
import data_generation
import networks
import scipy.io as sio
import param
import util
import truncated_vgg
from keras.backend.tensorflow_backend import set_session
from keras.optimizers import Adam
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import cv2

def train(model_name, gpu_id):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    set_session(tf.Session(config=config))

    with tf.Session() as sess:

        params = param.get_general_params()

        network_dir = params['model_save_dir'] + '/' + model_name

        # Creates models directory if not exist.
        if not os.path.isdir(network_dir):
            os.mkdir(network_dir)

        train_feed = data_generation.create_feed(params, params['data_dir'], 'train')
        # test_feed  = data_generation.create_feed(params, params['data_dir'], 'test')

        tf_config = tf.ConfigProto()
        tf_config.gpu_options.allow_growth = True
        tf_config.allow_soft_placement = True

        with tf.Session(config=tf_config) as sess:

            # Load VGG truncated model.
            vgg_model = truncated_vgg.vgg_norm()
            networks.make_trainable(vgg_model, False)

            # Load VGG pretrained weights.
            response_weights = sio.loadmat('../Models/vgg_activation_distribution_train.mat')

            # Create graph and compile keras model.
            model = networks.network_posewarp(params)
            tloss = networks.vgg_loss(vgg_model, response_weights, 12)
            model.compile(optimizer=Adam(lr=1e-4), loss=[tloss])

            # Get number of trainig steps.
            n_iters = params['n_training_iter']

            # Create a tensorboard writer.
            summary_writer = tf.summary.FileWriter("../logs/run2/", graph=sess.graph)

            tr_x, tr_y = next(train_feed)
            # te_x, te_y = next(test_feed)

            # Prepare output directories if they don't exist.
            output_dir = '../output/' + model_name + '/'

            if not os.path.isdir(output_dir):
                os.mkdir(output_dir)

            scipy.misc.imsave('../output/' + model_name + '/tr_orig_image.png', tr_x[0][0, :, :, :])
            scipy.misc.imsave('../output/' + model_name + '/tr_targ_image.png', tr_y[0, :, :, :])

            # Tensorboard logged tensors.
            gen = tf.get_default_graph().get_tensor_by_name("loss/add_2_loss/lambda_5/add:0")[0, :, :, :]
            inp = tf.get_default_graph().get_tensor_by_name("in_img0:0")[0, :, :, :]
            msk = tf.get_default_graph().get_tensor_by_name("mask_prior:0")[0, :, :, 0:1]
            msk = tf.tile(msk, [1, 1, 3])
            out = tf.get_default_graph().get_tensor_by_name("in_img1:0")[0, :, :, :]

            for step in range(0, n_iters):

                # Train with next batch.
                x, y = next(train_feed)

                train_loss = model.train_on_batch(x, y)

                # Print training loss progress.
                util.printProgress(step, 0, train_loss)

                # Add training loss to tensorboard.
                summary = tf.Summary()
                summary.value.add(tag='loss', simple_value=train_loss)
                summary_writer.add_summary(summary, step)

                if step % params['test_interval'] == 0:

                    # Set up tensorboard image summary.
                    image_summary_1 = tf.summary.image('images', [inp, msk, out, gen], max_outputs=100)

                    # Compute summary.
                    image_summary_1_run = sess.run(image_summary_1, feed_dict={"in_img0:0":    x[0],"in_pose0:0": x[1], "in_pose1:0": x[2],
                                                                               "mask_prior:0": x[3],"trans_in:0": x[4], "in_img1:0":  y})
                    # Register summary in tensorboard.
                    summary_writer.add_summary(image_summary_1_run)

                    # Compute training sample images.
                    train_image = sess.run(gen, feed_dict={"in_img0:0": tr_x[0],    "in_pose0:0": tr_x[1], "in_pose1:0": tr_x[2],
                                                           "mask_prior:0": tr_x[3], "trans_in:0": tr_x[4], "in_img1:0": tr_y})

                    # Save in disk computed sample images.
                    scipy.misc.imsave(output_dir + 'tr' + str(step) + ".png", train_image)


                # Save model checkpoints.
                if step > 0 and step % params['model_save_interval'] == 0:
                    model.save_weights(network_dir + '/' + str(step) + '.h5')

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Need model name and gpu id as command line arguments.")
    else:
        train(sys.argv[1], sys.argv[2])
