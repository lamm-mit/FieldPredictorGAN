#!/usr/bin/env python
# coding: utf-8

import os

PATH = './STITCH(GEO_MISES_1.2)/'
BUFFER_SIZE = 400
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256
OUTPUT_CHANNELS = 3
LAMBDA = 100
EPOCHS = 400
checkpoint_dir = './training_checkpoints'
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
