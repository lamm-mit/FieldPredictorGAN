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
if gpus:
  try:
    # Currently, memory growth needs to be the same across GPUs
    for gpu in gpus:
      tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
  except RuntimeError as e:
    # Memory growth must be set before GPUs have been initialized
    print(e)
print(gpus, cpus)
num_data = 400
filename = []
for i in range(num_data):
    filename.append(PATH+'test/' + str(i + 1) + ".png")
test_dataset = tf.data.Dataset.from_tensor_slices(filename)
#test_dataset = tf.data.Dataset.list_files(PATH+'test/*.png', shuffle=False)
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

checkpoint.restore(tf.train.latest_checkpoint(checkpoint_dir))

count = 0
for inp, tar in test_dataset.take(num_data):
  print(count)
  count += 1
  generate_images(generator, inp, tar, count)

