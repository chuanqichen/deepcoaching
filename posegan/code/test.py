# -*- coding: utf-8 -*-
import matplotlib.pyplot

import tensorflow as tf
import os
import sys
import data_generation
import networks
import scipy.io as sio
import param
import scipy.misc
import util
import truncated_vgg
import matplotlib.pyplot as plt
import numpy as np
from keras.backend.tensorflow_backend import set_session
from keras.optimizers import Adam

from os import listdir
from os.path import isfile, join
from datetime import datetime

def test(model_name, gpu_id):

    params = param.get_general_params()

    TEST_PATH = params['data_dir'] + "/test/test_golf/"

    SRC_IMG = TEST_PATH + "ref_img/"
    TGT_POS = TEST_PATH + "ref_pose/"

    with tf.Session() as sess:

        network_dir = params['model_save_dir'] + '/' + model_name

        # Creates models directory if not exist.
        if not os.path.isdir(network_dir):
            print("No model named ´" + model_name + "´ found!")
            raise

        img_feed = data_generation.create_feed(params, SRC_IMG, mode="test", do_augment=False)
        pos_feed = data_generation.create_feed(params, TGT_POS, mode="test", do_augment=False)

        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        set_session(tf.Session(config=config))

        vgg_model = truncated_vgg.vgg_norm()
        networks.make_trainable(vgg_model, False)
        response_weights = sio.loadmat('../data/vgg_activation_distribution_train.mat')

        ckp_name = [f for f in listdir(network_dir) if isfile(join(network_dir, f))][-1]

        model = networks.network_posewarp(params)
        model.compile(optimizer=Adam(lr=1e-4), loss=[networks.vgg_loss(vgg_model, response_weights, 12)])

        model.load_weights(network_dir + "/"  + ckp_name)

        n_iters = 1

        log_dir = "../logs/" + datetime.now().strftime("%H-%M")
        os.mkdir( log_dir )
        
        summary_writer = tf.summary.FileWriter(log_dir, graph=sess.graph)
        for step in range(0, n_iters):

            x_img = next(img_feed)
            x_pos = next(pos_feed)
            x_pos = next(pos_feed)
            x_pos = next(pos_feed)
            x_pos = next(pos_feed)

            gen = tf.get_default_graph().get_tensor_by_name("loss/add_2_loss/lambda_5/add:0")
            inp = tf.get_default_graph().get_tensor_by_name("in_img0:0")
            image_summary_op = tf.summary.image('images', [inp[0, :, :, :], gen[0, :, :, :]], max_outputs=100)
            image_summary = sess.run(image_summary_op, feed_dict={"in_img0:0" : x_img[0], "in_pose0:0" : x_img[1], "in_pose1:0" : x_pos[2], "mask_prior:0" : x_img[3], "trans_in:0" : x_img[4]})
            summary_writer.add_summary(image_summary)


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Need model name and gpu id as command line arguments.")
    else:
        test(sys.argv[1], sys.argv[2])
