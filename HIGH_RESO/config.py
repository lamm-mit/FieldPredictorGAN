#!/usr/bin/env python
# coding: utf-8

import os

PATH = './MIT/'
BUFFER_SIZE = 400
BATCH_SIZE = 1
IMG_WIDTH = 512
IMG_HEIGHT = 512
OUTPUT_CHANNELS = 3
LAMBDA = 100
EPOCHS = 300
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
