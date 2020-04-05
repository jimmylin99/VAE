# coding=utf-8
__version__ = 'v3'

from tensorflow.keras.layers import Dense, InputLayer, Concatenate
from tensorflow.keras.layers import Conv2D, Flatten, Lambda
from tensorflow.keras.layers import Reshape, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping, LambdaCallback
import datetime
import tensorflow as tf
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse
import os

"""
Description:
    Train model learning Duckietown environment, using VAE-like model.
LIN
2020/04/02
"""

# for the sake of running on hpc, which does not have display device
matplotlib.use('Agg')

parser = argparse.ArgumentParser()
parser.add_argument('-w', '--weights', help='Load tf model trained weights')
parser.add_argument('-e', '--epoch', help='epoch', default=2, type=int)
args = parser.parse_args()

# logs registry
log_dir = 'v4-logs/'
currentTimeStr = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
scalar_log_dir = os.path.join(log_dir, 'scalars/', currentTimeStr)
loss_file_writer = tf.summary.create_file_writer(scalar_log_dir + "/loss")
loss_file_writer.set_as_default()

# CONSTANT
TRAIN_VAL_SPLIT_CONSTANT = 0.8
# network parameters
# for distributed training, the batch_size (globally) will be divided into N parts for N GPUs.
# e.g. suppose batch for each GPU is 128, then batch_size = 512 for 4 GPUs, = 1024 for 8 GPUs.
# notice that larger batch often means more epochs to train
batch_size = 128
latent_dim = 20
epochs = args.epoch
loss_weight = {'image_loss': 10.0,
               'reward_loss': 1e-5,
               'done_loss': 1.0,
               'latent_loss': 1e-3}


class VAE(Model):
    def __init__(self):
        super(VAE, self).__init__()
        # VAE model = encoder + decoder
        # [input_four + action] --> [predicted_{img + reward + done}]
        # categorize input
        self.inputs = InputLayer(input_shape=input_shape, name='encoder_input')
        self.input_four = Lambda(lambda x:
                                 x[:, :image_size[0] - 1, :, :image_size[2] - 1])
        self.input_fifth = Lambda(lambda x:
                                  x[:, :image_size[0] - 1, :, image_size[2] - 1])
        # speed will not be counted in this version
        self.action = Lambda(lambda x:  # action is no longer two-value but a scalar, representing steering
                             x[:, image_size[0] - 1, 5:6, 0])
        self.reward = Lambda(lambda x:
                             x[:, image_size[0] - 1, 6:7, 0])
        self.done = Lambda(lambda x:  # done is a binary value, i.e. either 0.0 or 1.0
                           x[:, image_size[0] - 1, 7:8, 0])
        """
            Conv2D layer: Conv2D(filter_num, filter_size, activation, strides, padding)
            padding = 'valid': H = ceil((H1 - filter_Height + 1) / stride)
            padding = 'same':  H = ceil(H1 / stride)
        """
        # build encoder
        self.conv1 = Conv2D(12, (5, 5), activation='relu', strides=(2, 2), padding='valid')  # (?, 58, 78, 12)
        self.conv2 = Conv2D(24, (5, 5), activation='relu', strides=(2, 2), padding='valid')  # (?, 27, 37, 24)
        self.conv3 = Conv2D(36, (5, 5), activation='relu', strides=(2, 2), padding='valid')  # (?, 12, 17, 36)
        self.conv4 = Conv2D(48, (5, 5), activation='relu', strides=(1, 1), padding='valid')  # (?, 8, 13, 48)
        self.conv5 = Conv2D(48, (3, 3), activation='relu', strides=(1, 1), padding='valid')  # (?, 6, 11, 48)
        self.conv6 = Conv2D(64, (3, 3), activation='relu', strides=(1, 1), padding='valid')  # (?, 4, 9, 64)

        # FC layer
        self.flatten = Flatten()
        self.d1 = Dense(1000, activation='relu')  # (?, 1000)
        self.d2 = Dense(100, activation='relu')  # (?, 100)

        # activation function before sampling process is special
        # 'linear' for z_mean, clip at least from above in z_log_var in avoidance with NaN
        self.dmean = Dense(latent_dim, activation='linear', name='z_mean')
        self.dlogvar = Dense(latent_dim, activation='linear', name='z_log_var')

        self.sampling = Lambda(sampling, output_shape=(latent_dim,), name='z')

        # merge classical latent space z along with action
        self.merge = Concatenate()

        # build decoder model
        # 1. generate predicted_img
        self.dmerge1 = Dense(100, activation='relu')
        self.dmerge2 = Dense(1000, activation='relu')
        self.shape = [4, 9, 64]
        self.drecover = Dense(self.shape[0] * self.shape[1] * self.shape[2], activation='relu')
        self.reshape = Reshape((self.shape[0], self.shape[1], self.shape[2]))  # (?, 4, 9, 64)
        """
            Conv2DTranspose layer: (filter_num, filter_size, activation, strides, padding)
            padding == 'valid': H = (H1 - 1) * stride + filter_Height
            padding == 'same':  H = H1 * stride
        """
        self.deconv1 = Conv2DTranspose(48, (3, 3), activation='relu', strides=(1, 1), padding='valid')  # (?, 6, 11, 48)
        self.deconv2 = Conv2DTranspose(48, (3, 3), activation='relu', strides=(1, 1), padding='valid')  # (?, 8, 13, 48)
        self.deconv3 = Conv2DTranspose(36, (5, 5), activation='relu', strides=(1, 1),
                                       padding='valid')  # (?, 12, 17, 36)
        self.deconv4 = Conv2DTranspose(24, (5, 5), activation='relu', strides=(2, 2),
                                       padding='valid')  # (?, 27, 37, 24)
        self.deconv5 = Conv2DTranspose(12, (6, 6), activation='relu', strides=(2, 2),
                                       padding='valid')  # (?, 58, 78, 12)
        # activation for last step in each part of decoder is special
        # image will be clipped in [0, 1] as did for data set
        # reward will use 'linear'
        # done will use 'sigmoid'
        self.deconv6 = Conv2DTranspose(1, (6, 6),
                                       activation='sigmoid',
                                       strides=(2, 2),
                                       padding='valid')  # (?, 120, 160, 1)
        # predicted_img = Flatten()(predicted_img)  # (?, 120 * 160 * 1) ??necessary?

        # 2. generate predicted_reward
        self.dr1 = Dense(100, activation='relu')
        self.dr2 = Dense(1000, activation='relu')
        self.dr3 = Dense(1000, activation='relu')
        self.dr4 = Dense(100, activation='relu')
        self.dreward = Dense(1, activation='linear')  # (?, 1)

        # 3. generate predicted_done
        self.dd1 = Dense(100, activation='relu')
        self.dd2 = Dense(1000, activation='relu')
        self.dd3 = Dense(1000, activation='relu')
        self.dd4 = Dense(100, activation='relu')
        self.ddone = Dense(1, activation='sigmoid')  # (?, 1)

        self.input_var = None
        self.input_four_var = None
        self.input_fifth_var = None
        self.action_var = None
        self.reward_var = None
        self.done_var = None
        self.z_mean_var = None
        self.z_log_var_var = None
        self.predicted_img_var = None
        self.predicted_reward_var = None
        self.predicted_done_var = None

    def call(self, inputs, training=None, mask=None):
        self.input_var = self.inputs(inputs)
        self.input_four_var = self.input_four(self.input_var)
        self.input_fifth_var = self.input_fifth(self.input_var)
        self.action_var = self.action(self.input_var)
        self.reward_var = self.reward(self.input_var)
        self.done_var = self.done(self.input_var)

        x = self.input_four_var
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.conv6(x)
        x = self.flatten(x)
        x = self.d1(x)
        x = self.d2(x)

        self.z_mean_var = self.dmean(x)
        self.z_log_var_var = self.dlogvar(x)
        self.z_log_var_var = K.clip(self.z_log_var_var, -100.0, 10.0)  # ??
        z = self.sampling([self.z_mean_var, self.z_log_var_var])
        merged = self.merge([z, self.action_var])

        x = self.dmerge1(merged)
        x = self.dmerge2(x)
        x = self.drecover(x)
        x = self.reshape(x)

        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = self.deconv4(x)
        x = self.deconv5(x)
        self.predicted_img_var = self.deconv6(x)

        x = self.dr1(merged)
        x = self.dr2(x)
        x = self.dr3(x)
        x = self.dr4(x)
        self.predicted_reward_var = self.dreward(x)

        x = self.dd1(merged)
        x = self.dd2(x)
        x = self.dd3(x)
        x = self.dd4(x)
        self.predicted_done_var = self.ddone(x)

        return self.predicted_img_var, self.predicted_reward_var, self.predicted_done_var


optimizer = tf.keras.optimizers.Adam()

train_loss = tf.keras.metrics.Mean(name='train_loss')
train_reconstruction_loss = tf.keras.metrics.Mean(name='train_reconstruction_loss')
train_reward_loss = tf.keras.metrics.Mean(name='train_reward_loss')
train_kl_loss = tf.keras.metrics.Mean(name='train_kl_loss')
train_done_loss = tf.keras.metrics.Mean(name='train_done_loss')

test_loss = tf.keras.metrics.Mean(name='test_loss')
test_reconstruction_loss = tf.keras.metrics.Mean(name='test_reconstruction_loss')
test_reward_loss = tf.keras.metrics.Mean(name='test_reward_loss')
test_kl_loss = tf.keras.metrics.Mean(name='test_kl_loss')
test_done_loss = tf.keras.metrics.Mean(name='test_done_loss')


@tf.function
def train_step(inputs):
    with tf.GradientTape(persistent=False) as tape:
        predicted_img, predicted_reward, predicted_done = vae(inputs, training=True)
        _reconstruction_loss = MeanSquaredError()(predicted_img, vae.input_fifth_var)
        _reward_loss = MeanSquaredError()(predicted_reward, vae.reward_var)
        _kl_loss = -0.5 * (K.sum(
            1 - K.square(vae.z_mean_var)
            - K.exp(vae.z_log_var_var)
            + vae.z_log_var_var,
            axis=-1))
        _done_loss = BinaryCrossentropy()(predicted_done, vae.done_var)
        total_loss = loss_weight['image_loss'] * _reconstruction_loss + \
                     loss_weight['latent_loss'] * _kl_loss + \
                     loss_weight['reward_loss'] * _reward_loss + \
                     loss_weight['done_loss'] * _done_loss

    gradients = tape.gradient(total_loss, vae.trainable_variables)
    optimizer.apply_gradients(zip(gradients, vae.trainable_variables))

    # reward_trainable_vairables = vae.dr1.trainable_variables + vae.dr2.trainable_variables + \
    #                              vae.dr3.trainable_variables + vae.dr4.trainable_variables + \
    #                              vae.dreward.trainable_variables
    # print(reward_trainable_vairables)
    # gradients_reward = tape.gradient(_reward_loss, reward_trainable_vairables)
    # optimizer.apply_gradients((zip(gradients_reward, reward_trainable_vairables)))

    train_loss(total_loss)
    train_reward_loss(_reward_loss)
    train_reconstruction_loss(_reconstruction_loss)
    train_kl_loss(_kl_loss)
    train_done_loss(_done_loss)


@tf.function
def test_step(inputs):
    predicted_img, predicted_reward, predicted_done = vae(inputs, training=True)
    _reconstruction_loss = MeanSquaredError()(predicted_img, vae.input_fifth_var)
    _reward_loss = MeanSquaredError()(predicted_reward, vae.reward_var)
    _kl_loss = -0.5 * (K.sum(
        1 - K.square(vae.z_mean_var)
        - K.exp(vae.z_log_var_var)
        + vae.z_log_var_var,
        axis=-1))
    _done_loss = BinaryCrossentropy()(predicted_done, vae.done_var)
    total_loss = loss_weight['image_loss'] * _reconstruction_loss + \
                 loss_weight['latent_loss'] * _kl_loss + \
                 loss_weight['reward_loss'] * _reward_loss + \
                 loss_weight['done_loss'] * _done_loss

    test_loss(total_loss)
    test_reconstruction_loss(_reconstruction_loss)
    test_reward_loss(_reward_loss)
    test_kl_loss(_kl_loss)
    test_done_loss(_done_loss)


@tf.function
def sampling(args):
    """
    Re-parametrization trick by sampling fr an isotropic unit Gaussian.
    :param args: (tensor) mean and log of variance of Q(z|X)
    :return: z (tensor): sampled latent vector
    """
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


# load DUCKIETOWN data set
dataset_file = np.load('dataset_vae.npz')
data_set = dataset_file["arr_0"]
# data pre-processing
# the following operation maps image value to [0, 1], without affecting value of
# other parameters, i.e. actions, reward, etc.
# notice that type of all values is the same, i.e. 'float32'
image_size = data_set.shape[1:]
input_shape = (image_size[0], image_size[1], image_size[2])
data_set = data_set.astype('float32') / 255
data_set[:, image_size[0] - 1, :, :] *= 255
print(data_set[10, :, :, :])
input('Press ENTER to continue...')

# shuffle the data set
np.random.shuffle(data_set)
# split data set into training and validation sets
# notice that there is no definition on y_train or y_val
x_train, x_val = np.split(data_set, [math.floor(TRAIN_VAL_SPLIT_CONSTANT * data_set.shape[0])])
print('x_train''s shape is: ' + str(x_train.shape))

train_ds = tf.data.Dataset.from_tensor_slices(x_train).shuffle(10000).batch(batch_size)

test_ds = tf.data.Dataset.from_tensor_slices(x_val).batch(batch_size)

vae = VAE()
# vae.summary()
# plot_model(vae, to_file='vae_cnn.png', show_shapes=True)

for epoch in range(epochs):
    train_loss.reset_states()
    train_reconstruction_loss.reset_states()
    train_reward_loss.reset_states()
    train_kl_loss.reset_states()
    train_done_loss.reset_states()

    test_loss.reset_states()
    test_reconstruction_loss.reset_states()
    test_reward_loss.reset_states()
    test_kl_loss.reset_states()
    test_done_loss.reset_states()

    for batch_inputs in train_ds:
        # print(batch_inputs.shape)
        train_step(batch_inputs)

    for batch_inputs in test_ds:
        print(batch_inputs.shape)
        test_step(batch_inputs)

    template = 'Epoch {}, Loss: {}, reconstruction loss: {}, reward loss: {}, kl loss: {}, done loss: {}\n' \
               '---- Test Loss: {}, reconstruction loss: {}, reward loss: {}, kl loss: {}, done loss: {}'
    print(template.format(epoch + 1,
                          train_loss.result(),
                          train_reconstruction_loss.result(),
                          train_reward_loss.result(),
                          train_kl_loss.result(),
                          train_done_loss.result(),
                          test_loss.result(),
                          test_reconstruction_loss.result(),
                          test_reward_loss.result(),
                          test_kl_loss.result(),
                          test_done_loss.result()))

    tf.summary.scalar(name='train/total_loss', data=train_loss.result(), step=epoch)
    tf.summary.scalar(name='train/reconstruction_loss', data=train_reconstruction_loss.result(), step=epoch)
    tf.summary.scalar(name='train/reward_loss', data=train_reward_loss.result(), step=epoch)
    tf.summary.scalar(name='train/kl_loss', data=train_kl_loss.result(), step=epoch)
    tf.summary.scalar(name='train/done_loss', data=train_done_loss.result(), step=epoch)
    tf.summary.scalar(name='test/total_loss', data=test_loss.result(), step=epoch)
    tf.summary.scalar(name='test/reconstruction_loss', data=test_reconstruction_loss.result(), step=epoch)
    tf.summary.scalar(name='test/reward_loss', data=test_reward_loss.result(), step=epoch)
    tf.summary.scalar(name='test/kl_loss', data=test_kl_loss.result(), step=epoch)
    tf.summary.scalar(name='test/done_loss', data=test_done_loss.result(), step=epoch)

tf.summary.scalar(name='loss_weight/kl', data=loss_weight['latent_loss'], step=1)
tf.summary.scalar(name='loss_weight/done', data=loss_weight['done_loss'], step=1)
tf.summary.scalar(name='loss_weight/reconstruction', data=loss_weight['image_loss'], step=1)
tf.summary.scalar(name='loss_weight/reward', data=loss_weight['reward_loss'], step=1)
