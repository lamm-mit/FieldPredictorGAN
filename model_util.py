#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import time
import PIL.Image as Image
from matplotlib import pyplot as plt
from IPython import display
from config import *
from model import *


#generator = Generator()
#tf.keras.utils.plot_model(generator, show_shapes=True, dpi=64)

#discriminator = Discriminator()
#tf.keras.utils.plot_model(discriminator, show_shapes=True, dpi=64)

#generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
#discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

#checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
#                                 discriminator_optimizer=discriminator_optimizer,
#                                 generator=generator,
#                                 discriminator=discriminator)


#import datetime
#log_dir="logs/"

#summary_writer = tf.summary.create_file_writer(
#  log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))



def generate_images(model, test_input, tar, count):
  prediction = model(test_input, training=True)
  plt.figure(figsize=(25, 25))
  #plt.figure(figsize=(6.5, 6.5))

  #plt.imshow(tar[0] * 0.5 + 0.5)
  #plt.axis('off')
  #plt.savefig('./predict/' + str(count) + '.png', bbox_inches='tight',pad_inches=-0.1)
  display_list = [test_input[0], tar[0], prediction[0]]
  #title = ['Input Image', 'Ground Truth', 'Predicted Image']


  for i in range(3):
    plt.subplot(1, 3, i+1)
    #plt.title(title[i])
    # getting the pixel values between [0, 1] to plot it.
    plt.imshow(display_list[i] * 0.5 + 0.5)
    plt.axis('off')
  plt.savefig('./predict/' + str(count) + '.png')

def predict_images(model, test_input):
  prediction = model(test_input, training=True)
  display_list = [test_input[0], prediction[0]]
  to_image = Image.new('RGB', (IMG_WIDTH * 2, IMG_WIDTH))
  for i in range(2):
    # getting the pixel values between [0, 1] to plot it.
    img_temp = tf.keras.preprocessing.image.array_to_img(display_list[i] * 0.5 + 0.5)
    to_image.paste(img_temp, (i * IMG_WIDTH, 0))
  return to_image


    

  
@tf.function
def train_step(input_image, target, epoch,  generator, discriminator, generator_optimizer, discriminator_optimizer, summary_writer):
  
  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    gen_output = generator(input_image, training=True)

    disc_real_output = discriminator([input_image, target], training=True)
    disc_generated_output = discriminator([input_image, gen_output], training=True)

    gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
    disc_loss = discriminator_loss(disc_real_output, disc_generated_output)

  generator_gradients = gen_tape.gradient(gen_total_loss,
                                          generator.trainable_variables)
  discriminator_gradients = disc_tape.gradient(disc_loss,
                                               discriminator.trainable_variables)

  generator_optimizer.apply_gradients(zip(generator_gradients,
                                          generator.trainable_variables))
  discriminator_optimizer.apply_gradients(zip(discriminator_gradients,
                                              discriminator.trainable_variables))
  with summary_writer.as_default():
    tf.summary.scalar('gen_total_loss', gen_total_loss, step=epoch)
    tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=epoch)
    tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=epoch)
    tf.summary.scalar('disc_loss', disc_loss, step=epoch)

def fit(train_ds, epochs, test_ds, generator, discriminator, generator_optimizer, discriminator_optimizer, checkpoint, summary_writer):
  for epoch in range(epochs):
    start = time.time()

    display.clear_output(wait=True)

    #for example_input, example_target in test_ds.take(1):
    #  generate_images(generator, example_input, example_target)
    print("Epoch: ", epoch)

    # Train
    for n, (input_image, target) in train_ds.enumerate():
      print('.', end='')
      if (n+1) % 100 == 0:
        print()
      train_step(input_image, target, epoch, generator, discriminator, generator_optimizer, discriminator_optimizer, summary_writer)
    print()

    # saving (checkpoint) the model every 20 epochs
    if (epoch + 1) % 20 == 0:
      checkpoint.save(file_prefix = checkpoint_prefix)

    print ('Time taken for epoch {} is {} sec\n'.format(epoch + 1,
                                                        time.time()-start))
  checkpoint.save(file_prefix = checkpoint_prefix)
