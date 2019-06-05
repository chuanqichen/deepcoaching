import tensorflow as tf
import os
import numpy as np
import sys
import data_generation
import networks
import scipy.io as sio
import param
import util
import truncated_vgg

import sys
import scipy
import time
from keras.optimizers import Adam
from tensorflow.python.client import timeline
import matplotlib.pyplot as plt

def train(model_name, gpu_id):

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    tf_config.allow_soft_placement = True

    with tf.Session(config=tf_config) as sess:

        params = param.get_general_params()
        network_dir = params['model_save_dir'] + '/' + model_name

        if not os.path.isdir(network_dir):
            os.mkdir(network_dir)

        train_feed = data_generation.create_feed(params, params['data_dir'], "train")
        # test_feed = data_generation.create_feed(params,  params['data_dir'], "test")

        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

        gan_lr  = 1e-3
        disc_lr = 1e-3
        disc_loss = 0.1

        generator = networks.network_posewarp(params)
        # generator.load_weights('../models/posewarp_vgg/100000.h5')

        discriminator = networks.discriminator(params)
        discriminator.compile(loss='binary_crossentropy', optimizer=Adam(lr=disc_lr))

        vgg_model = truncated_vgg.vgg_norm()
        networks.make_trainable(vgg_model, False)
        response_weights = sio.loadmat('../data/vgg_activation_distribution_train.mat')

        gan = networks.gan(generator, discriminator, params)

        gan.compile(optimizer=Adam(lr=gan_lr),
                    loss=[networks.vgg_loss(vgg_model, response_weights, 12), 'binary_crossentropy'],
                    loss_weights=[1.0, disc_loss])

        n_iters = params['n_training_iter']
        batch_size = params['batch_size']

        summary_writer = tf.summary.FileWriter("../logs", graph=sess.graph)

        tr_x, tr_y = next(train_feed)
        # te_x, te_y = next(test_feed)

        # Prepare output directories if they don't exist.
        output_dir = '../output/' + model_name + '/'

        if not os.path.isdir(output_dir):
            os.mkdir(output_dir)

        scipy.misc.imsave(output_dir + 'tr_orig_image.png', tr_x[0][0, :, :, :])
        scipy.misc.imsave(output_dir + 'tr_targ_image.png', tr_y[0, :, :, :])
        # scipy.misc.imsave(output_dir + 'te_orig_image.png', te_x[0][0, :, :, :])
        # scipy.misc.imsave(output_dir + 'te_targ_image.png', te_y[0, :, :, :])

        print("Batch size: " + str(batch_size))

        for step in range(n_iters):

            x, y = next(train_feed)

            gen = generator.predict(x)

            # Train discriminator
            x_tgt_img_disc  = np.concatenate((y, gen))
            x_src_pose_disc = np.concatenate((x[1], x[1]))
            x_tgt_pose_disc = np.concatenate((x[2], x[2]))

            L = np.zeros([2 * batch_size])
            L[0:batch_size] = 1

            inputs = [x_tgt_img_disc, x_src_pose_disc, x_tgt_pose_disc]
            d_loss = discriminator.train_on_batch(inputs, L)

            # Train the discriminator a couple of iterations before starting the gan
            if step < 5:
                util.printProgress(step, 0, [0, d_loss])
                step += 1
                continue

            # TRAIN GAN
            L = np.ones([batch_size])
            x, y = next(train_feed)
            g_loss = gan.train_on_batch(x, [y, L])
            util.printProgress(step, 0, [g_loss[1], d_loss])

            if step % params['test_interval'] == 0:

                print(gen[0])

                gen = tf.get_default_graph().get_tensor_by_name("model_1/add_2_1/add:0")
                inp = tf.get_default_graph().get_tensor_by_name("in_img0:0")
                out = tf.get_default_graph().get_tensor_by_name("in_img1:0")
                p_s = tf.get_default_graph().get_tensor_by_name("mask_src/truediv:0")
                # p_t = tf.get_default_graph().get_tensor_by_name("in_pose1:0")


                image_summary_1 = tf.summary.image('images', [inp[0, :, :, :], out[0, :, :, :], gen[0, :, :, :]], max_outputs=100)
                # image_summary_2 = tf.summary.image('pose', [tf.reduce_sum(p_s[0, :, :, :], 2, keepdims=True)], max_outputs=100)

                image_summary_1 = sess.run(image_summary_1,feed_dict={"in_img0:0": x[0], "in_pose0:0": x[1], "in_pose1:0": x[2],
                                                                      "mask_prior:0": x[3], "trans_in:0": x[4], "in_img1:0": y,
                                                                      "input_3:0": x[0], "input_4:0": x[1], "input_5:0": x[2],
                                                                      "input_6:0": x[3], "input_7:0": x[4]})
                #
                # img_gen =  sess.run(image_summary_1,feed_dict={"in_img0:0": x[0], "in_pose0:0": x[1], "in_pose1:0": x[2],
                #                                                "mask_prior:0": x[3], "trans_in:0": x[4], "in_img1:0": y,
                #                                                "input_3:0": x[0], "input_4:0": x[1], "input_5:0": x[2],
                #                                                "input_6:0": x[3], "input_7:0": x[4]})


                # image_summary_2 = sess.run(image_summary_2, feed_dict={"in_img0:0" : x[0], "in_pose0:0" : x[1], "in_pose1:0" : x[2],
                #                                                     "mask_prior:0" : x[3], "trans_in:0" : x[4], "in_img1:0"  : y})

                summary_writer.add_summary(image_summary_1)
                # summary_writer.add_summary(image_summary_2)

                train_image = sess.run(gen, feed_dict={"in_img0:0": tr_x[0], "in_pose0:0": tr_x[1], "in_pose1:0": tr_x[2],
                                                       "mask_prior:0": tr_x[3], "trans_in:0": tr_x[4], "in_img1:0": tr_y,
                                                       "input_3:0": tr_x[0], "input_4:0": tr_x[1], "input_5:0": tr_x[2],
                                                       "input_6:0": tr_x[3], "input_7:0": tr_x[4]})
                #
                # test_image = sess.run(gen, feed_dict={"in_img0:0": te_x[0], "in_pose0:0": te_x[1], "in_pose1:0": te_x[2],
                #                                       "mask_prior:0": te_x[3], "trans_in:0": te_x[4], "in_img1:0": te_y,
                #                                       "input_3:0": te_x[0], "input_4:0": te_x[1], "input_5:0": te_x[2],
                #                                       "input_6:0": te_x[3], "input_7:0": te_x[4]})


                scipy.misc.imsave(output_dir + 'tr' + str(step) + ".png", train_image[0, :, :, :])
                # scipy.misc.imsave(output_dir + 'te' + str(step) + ".png", test_image[0, :, :, :])

            if step % params['model_save_interval'] == 0 and step > 0:
                gan.save(network_dir + '/' + str(step) + '.h5')


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Need model name and gpu id as command line arguments.")
    else:
        train(sys.argv[1], sys.argv[2])
