# coding=utf-8
__version__ = 'v3'

from tensorflow.keras.layers import Dense, Input, concatenate
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

# CONSTANT
TRAIN_VAL_SPLIT_CONSTANT = 0.8
# network parameters
# for distributed training, the batch_size (globally) will be divided into N parts for N GPUs.
# e.g. suppose batch for each GPU is 128, then batch_size = 512 for 4 GPUs, = 1024 for 8 GPUs.
# notice that larger batch often means more epochs to train
batch_size = 128
latent_dim = 20
epochs = args.epoch
loss_weight = {'image_loss': 1.0,
               'reward_loss': 1.0,
               'done_loss': 1.0,
               'latent_loss': 1.0}


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


# custom activation function should use Keras backend (K), so that features like auto gradient could work
def clip_activation(x, x_min, x_max):
    return K.clip(x, x_min, x_max)


# custom loss function
def custom_loss(y_true, y_pred):
    # notice that MeanSquaredError is different from mse, due to their inconsistent usage of K.mean
    # class MeanSquaredError, BinaryCrossentropy: their __call__ method will return a scalar
    # which coincides with the return type of custom_loss function, i.e. scalar
    # to be more precise: (tensor) shape=()
    image_loss = MeanSquaredError()(y_true=input_fifth, y_pred=predicted_img)  # scalar
    reward_loss = MeanSquaredError()(y_true=reward, y_pred=predicted_reward)  # scalar
    done_loss = BinaryCrossentropy()(y_true=done, y_pred=predicted_done)  # scalar
    # notice that latent_loss is related to latent_dim
    # here we divided it by latent_dim
    latent_loss = 1 - K.square(z_mean) - K.exp(z_log_var) + z_log_var  # (?, latent_dim)
    latent_loss = K.sum(latent_loss, axis=-1)  # sum along the last axis --> (?,)
    latent_loss *= -0.5
    # make latent_loss irrelevant to latent_dim, where the latter one may vary as a hyper-parameter
    latent_loss /= latent_dim
    latent_loss = K.mean(latent_loss)  # take mean over batch --> scalar

    overall_loss = \
        loss_weight['image_loss'] * image_loss + \
        loss_weight['reward_loss'] * reward_loss + \
        loss_weight['done_loss'] * done_loss + \
        loss_weight['latent_loss'] * latent_loss
    return overall_loss


# load DUCKIETOWN data set
dataset_file = np.load('dataset_vae.npz')
data_set = dataset_file["arr_0"]
# data pre-processing
# the following operation maps image value to [0, 1], without affecting value of
# other parameters, i.e. actions, reward, etc.
# notice that type of all values is the same, i.e. 'float32'
image_size = data_set.shape[1:]
data_set = data_set.astype('float32') / 255
data_set[:, :image_size[0] - 1, :, :] *= 255
print(data_set[10, :, :, :])
input('Press ENTER to continue...')

# shuffle the data set
np.random.shuffle(data_set)
# split data set into training and validation sets
# notice that there is no definition on y_train or y_val
x_train, x_val = np.split(data_set, [math.floor(TRAIN_VAL_SPLIT_CONSTANT * data_set.shape[0])])
print('x_train''s shape is: ' + str(x_train.shape))

# VAE model = encoder + decoder
# [input_four + action] --> [predicted_{img + reward + done}]
# categorize input
input_shape = (image_size[0], image_size[1], image_size[2])
inputs = Input(shape=input_shape, name='encoder_input')
input_four = Lambda(lambda x:
                    x[:, :image_size[0] - 1, :, :image_size[2] - 1])(inputs)
input_fifth = Lambda(lambda x:
                     x[:, :image_size[0] - 1, :, image_size[2] - 1])(inputs)
# speed will not be counted in this version
action = Lambda(lambda x:  # action is no longer two-value but a scalar, representing steering
                    x[:, image_size[0] - 1, 5:6, 0])(inputs)
reward = Lambda(lambda x:
                  x[:, image_size[0] - 1, 6:7, 0])(inputs)
done = Lambda(lambda x:  # done is a binary value, i.e. either 0.0 or 1.0
                  x[:, image_size[0] - 1, 7:8, 0])(inputs)
"""
    Conv2D layer: Conv2D(filter_num, filter_size, activation, strides, padding)
    padding = 'valid': H = ceil((H1 - filter_Height + 1) / stride)
    padding = 'same':  H = ceil(H1 / stride)
"""
# build encoder
x = input_four  # (?, 120, 160, 4)
x = Conv2D(12, (5, 5), activation='relu', strides=(2, 2), padding='valid')(x)  # (?, 58, 78, 12)
x = Conv2D(24, (5, 5), activation='relu', strides=(2, 2), padding='valid')(x)  # (?, 27, 37, 24)
x = Conv2D(36, (5, 5), activation='relu', strides=(2, 2), padding='valid')(x)  # (?, 12, 17, 36)
x = Conv2D(48, (5, 5), activation='relu', strides=(1, 1), padding='valid')(x)  # (?, 8, 13, 48)
x = Conv2D(48, (3, 3), activation='relu', strides=(1, 1), padding='valid')(x)  # (?, 6, 11, 48)
x = Conv2D(64, (3, 3), activation='relu', strides=(1, 1), padding='valid')(x)  # (?, 4, 9, 64)

# shape info needed to build decoder model
shape = K.int_shape(x)

# FC layer
x = Flatten()(x)
x = Dense(1000, activation='relu')(x)  # (?, 1000)
x = Dense(100, activation='relu')(x)  # (?, 100)

# activation function before sampling process is special
# 'linear' for z_mean, clip at least from above in z_log_var in avoidance with NaN
z_mean = Dense(latent_dim, activation='linear', name='z_mean')(x)
z_log_var = Dense(latent_dim, activation=clip_activation(-100.0, 10.0), name='z_log_var')(x)

z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# merge classical latent space z along with action
merged = concatenate([z, action])

# build decoder model
# 1. generate predicted_img
x = Dense(100, activation='relu')(merged)
x = Dense(1000, activation='relu')(x)
x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(x)
x = Reshape((shape[1], shape[2], shape[3]))(x)  # (?, 4, 9, 64)
"""
    Conv2DTranspose layer: (filter_num, filter_size, activation, strides, padding)
    padding == 'valid': H = (H1 - 1) * stride + filter_Height
    padding == 'same':  H = H1 * stride
"""
x = Conv2DTranspose(48, (3, 3), activation='relu', strides=(1, 1), padding='valid')(x)  # (?, 6, 11, 48)
x = Conv2DTranspose(48, (3, 3), activation='relu', strides=(1, 1), padding='valid')(x)  # (?, 8, 13, 48)
x = Conv2DTranspose(36, (5, 5), activation='relu', strides=(1, 1), padding='valid')(x)  # (?, 12, 17, 36)
x = Conv2DTranspose(24, (5, 5), activation='relu', strides=(2, 2), padding='valid')(x)  # (?, 27, 37, 24)
x = Conv2DTranspose(12, (6, 6), activation='relu', strides=(2, 2), padding='valid')(x)  # (?, 58, 78, 12)
# activation for last step in each part of decoder is special
# image will be clipped in [0, 1] as did for data set
# reward will use 'linear'
# done will use 'sigmoid'
predicted_img = Conv2DTranspose(1, (6, 6),
                                activation=clip_activation(x_min=0.0, x_max=1.0),
                                strides=(2, 2),
                                padding='valid')(x)   # (?, 120, 160, 1)
# predicted_img = Flatten()(predicted_img)  # (?, 120 * 160 * 1) ??necessary?

# 2. generate predicted_reward
x = Dense(100, activation='relu')(merged)
x = Dense(1000, activation='relu')(x)
x = Dense(1000, activation='relu')(x)
x = Dense(100, activation='relu')(x)
predicted_reward = Dense(1, activation='linear')(x)  # (?, 1)

# 3. generate predicted_done
x = Dense(100, activation='relu')(merged)
x = Dense(1000, activation='relu')(x)
x = Dense(1000, activation='relu')(x)
x = Dense(100, activation='relu')(x)
predicted_done = Dense(1, activation='sigmoid')(x)  # (?, 1)

outputs = [predicted_img, predicted_reward, predicted_done]

# create and compile VAE model which can be run in a distributional way across multiple GPUs.
mirrored_strategy = tf.distribute.MirroredStrategy()
with mirrored_strategy.scope():
    vae = Model(inputs, outputs, name='vae')
    # use custom loss
    vae.compile(optimizer='adam', loss=custom_loss)

vae.summary()
plot_model(vae, to_file='vae_cnn.png', show_shapes=True)
tf.keras.callbacks.Callback()
# logs registry
log_dir = 'logs/'
currentTimeStr = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
scalar_log_dir = os.path.join(log_dir, 'scalars/', currentTimeStr)
loss_file_writer = tf.summary.create_file_writer(scalar_log_dir + "/loss")
loss_file_writer.set_as_default()

fit_log_dir = os.path.join(log_dir, 'fit/', currentTimeStr)
weight_dir = os.path.join(
    "weight",
    currentTimeStr,
)
# model_dir = os.path.join(
#     "model",
#     datetime.datatime.now().strftime("%Y%m%d-%H%M%S"),
#
# )
tbCallBack = TensorBoard(log_dir=log_dir,  # log 目录
                         histogram_freq=0,  # 按照何等频率（epoch）来计算直方图，0为不计算
                         #                  batch_size=32,     # 用多大量的数据计算直方图
                         write_graph=True,  # 是否存储网络结构图
                         write_grads=True,  # 是否可视化梯度直方图
                         write_images=True,  # 是否可视化参数
                         embeddings_freq=0,
                         embeddings_layer_names=None,
                         embeddings_metadata=None,
                         # update_freq=100
                         )

cpCallBack = ModelCheckpoint(weight_dir,
                             monitor='val_loss',
                             verbose=0,
                             save_best_only=True,
                             save_weights_only=True,
                             mode='min',
                             save_freq='epoch')

esCallBack = EarlyStopping(monitor='val_loss',
                           min_delta=0,
                           patience=10,
                           verbose=0,
                           mode='min',
                           baseline=None,
                           restore_best_weights=False) # ???
from tensorflow.keras.models import load_model
if args.weights:
    vae.load_weights(args.weights)
    vae = load_model('vae_cnn_model.tf')
    vae.fit(x_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_test, y_test),
            callbacks=[tbCallBack, cpCallBack, esCallBack])
    vae.save_weights('vae_cnn_duckie.tf')
    vae.save('vae_cnn_model.tf')
else:
    # train the autoencoder
    vae.fit(x_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_test, y_test),
            callbacks=[tbCallBack, cpCallBack, esCallBack])
    # weight recorded at the last step, could be different from the one in callback
    vae.save_weights('vae_cnn_duckie.tf')
    vae.save('vae_cnn_model.tf')

print('loading best weights...')
vae.load_weights(weight_dir)
print('producing sampled results...')
# plot_results(models, data, batch_size=batch_size, model_name="vae_cnn")
outputImg = vae.predict(x_test, batch_size=batch_size)
figure = plt.figure()
ax = figure.add_subplot(221)
ax.title.set_text('Ground Truth (Validation Data Set)')
ax.imshow(np.reshape(y_test[0], [image_size[0] - 1, image_size[1]]), cmap=plt.cm.gray)
ax = figure.add_subplot(222)
ax.title.set_text('Prediction (Validation Data Set)')
ax.imshow(np.reshape(outputImg[0], [image_size[0] - 1, image_size[1]]), cmap=plt.cm.gray)
ax = figure.add_subplot(223)
ax.title.set_text('Ground Truth (Training Data Set)')
ax.imshow(np.reshape(y_train[0], [image_size[0] - 1, image_size[1]]), cmap=plt.cm.gray)
ax = figure.add_subplot(224)
ax.title.set_text('Prediction (Training Data Set)')
ax.imshow(np.reshape(vae.predict(x_train, batch_size=batch_size)[0],
                     [image_size[0] - 1, image_size[1]]),
          cmap=plt.cm.gray)
# plt.show()
plt.subplots_adjust(wspace=0.4, hspace=0.5)
plt.savefig('showSampledResults.png')
print('completed')
# figure = plt.figure()
# plt.imshow(figure, cmap='Greys_r')
