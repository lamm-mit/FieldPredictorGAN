#!/usr/bin/env python
# coding: utf-8

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import os
import time

from matplotlib import pyplot as plt
from IPython import display
from datatool import *
from model import *
from model_util import *
from config import *

gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
cpus = tf.config.experimental.list_physical_devices(device_type='CPU')
#print("yes")
tf.config.experimental.set_memory_growth(gpus[0], True)
print(gpus, cpus)


train_dataset = tf.data.Dataset.list_files(PATH+'train/*.png')
train_dataset = train_dataset.map(load_image_train,
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE)
train_dataset = train_dataset.shuffle(BUFFER_SIZE)
train_dataset = train_dataset.batch(BATCH_SIZE)
test_dataset = tf.data.Dataset.list_files(PATH+'test/*.png')
test_dataset = test_dataset.map(load_image_test)
test_dataset = test_dataset.batch(BATCH_SIZE)

generator = Generator()
#tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64)

discriminator = Discriminator()
#tf.keras.utils.plot_model(discriminator, show_shapes=True, dpi=64)

generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)
#checkpoint.restore(tf.train.latest_checkpoint(checkpoint_prefix))


import datetime
log_dir="logs/"

summary_writer = tf.summary.create_file_writer(
  log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

fit(train_dataset, EPOCHS, test_dataset, generator, discriminator, generator_optimizer, discriminator_optimizer, checkpoint, summary_writer)
