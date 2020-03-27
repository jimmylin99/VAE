from tensorflow.keras.layers import Dense, Input, concatenate
from tensorflow.keras.layers import Conv2D, Flatten, Lambda
from tensorflow.keras.layers import Reshape, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras.losses import mse, binary_crossentropy
from tensorflow.keras.utils import plot_model
from tensorflow.keras import backend as K
import tensorflow as tf
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import os


parser = argparse.ArgumentParser()
help_ = "Load h5 model trained weights"
parser.add_argument("-w", "--weights", help=help_)
help_ = "Use mse loss instead of binary cross entropy (default)"
parser.add_argument("-m", "--mse", help=help_, action='store_true')
parser.add_argument("-e", "--epoch", help="epoch", default=2, type=int)
args = parser.parse_args()

# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# then z = z_mean + sqrt(var)*eps
def sampling(args):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.
    # Arguments
        args (tensor): mean and log of variance of Q(z|X)
    # Returns
        z (tensor): sampled latent vector
    """
    z_mean, z_log_var = args
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


# DUCKIETOWN dataset
dataset_file = np.load('dataset_vae.npz')
dataset = dataset_file["arr_0"]

np.random.shuffle(dataset)
x_train, x_test = np.split(dataset, [math.floor(0.9 * dataset.shape[0])])
print(x_train.shape)
image_size = [x_train.shape[1], x_train.shape[2], x_train.shape[3]]
x_train = np.reshape(x_train, [-1, image_size[0], image_size[1], image_size[2]])
x_test = np.reshape(x_test, [-1, image_size[0], image_size[1], image_size[2]])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255
y_train = x_train[:, :image_size[0] - 1, :, image_size[2] - 1]
y_test = x_test[:, :image_size[0] - 1, :, image_size[2] - 1]

# network parameters
input_shape = (image_size[0], image_size[1], image_size[2])
batch_size = 128
kernel_size = 5
filters = [16, 32, 64]
latent_dim = 10
epochs = args.epoch
strides = (2, 2)

# VAE model = encoder + decoder
# build encoder model
inputs = Input(shape=input_shape, name='encoder_input')
input_four = Lambda(lambda x:
                    x[:, :image_size[0] - 1, :, :image_size[2] - 1])(inputs)
input_fifth = Lambda(lambda x:
                     x[:, :image_size[0] - 1, :, image_size[2] - 1])(inputs)
actions = Lambda(lambda x:
                    x[:, image_size[0] - 1, 0:2, 0])(inputs)
reward = Lambda(lambda x:
                  x[:, image_size[0] - 1, 2:3, 0])(inputs)
done_int = Lambda(lambda x:
                  x[:, image_size[0] - 1, 3:4, 0])(inputs)
speed = Lambda(lambda x:
                  x[:, image_size[0] - 1, 4:5, 0])(inputs)
x = input_four
for f in filters:
    x = Conv2D(filters=f,
               kernel_size=kernel_size,
               activation='relu',
               strides=strides,
               padding='same')(x)

# shape info needed to build decoder model
shape = K.int_shape(x)

# generate latent vector Q(z|X)
x = Flatten()(x)
x = Dense(latent_dim, activation='relu')(x)


def clip_activation(x_min=-20.0, x_max=10.0):
    def wrapped_activation(x):
        return K.clip(x, x_min, x_max)
    return wrapped_activation


z_mean = Dense(latent_dim, activation=clip_activation(-10000.0, 10000.0), name='z_mean')(x)
z_log_var = Dense(latent_dim, activation=clip_activation(-20.0, 10.0), name='z_log_var')(x)
# z_mean = Dense(latent_dim, activation='sigmoid', name='z_mean')(x)
# z_log_var = Dense(latent_dim, activation='sigmoid', name='z_log_var')(x)

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# instantiate encoder model
# encoder = Model(inputs, [z, actions, speed], name='encoder')
# encoder.summary()
# plot_model(encoder, to_file='vae_cnn_encoder.png', show_shapes=True)

# Concatenate
# first_input = Input(shape=(latent_dim,))
# second_input = Input(shape=(2,))
# third_input = Input(shape=(1,))
# merged = concatenate([first_input, second_input, third_input])
# print(merged)
# merged_model = Model([first_input, second_input, third_input], merged, name='merged_model')

merged = concatenate([z, actions, speed])

# build decoder model
# latent_inputs = Input(shape=(latent_dim + 3,), name='z_sampling')
x = Dense(shape[1] * shape[2] * shape[3], activation='relu')(merged)
x = Reshape((shape[1], shape[2], shape[3]))(x)

for f in filters[::-1]:
    x = Conv2DTranspose(filters=f,
                        kernel_size=kernel_size,
                        activation='relu',
                        strides=strides,
                        padding='same')(x)
# clip_activation
outputs = Conv2DTranspose(filters=1,
                          kernel_size=kernel_size,
                          activation='sigmoid',
                          padding='same',
                          strides=(1, 1),
                          name='decoder_output')(x)

# instantiate decoder model
# decoder = Model(latent_inputs, outputs, name='decoder')
# decoder.summary()
# plot_model(decoder, to_file='vae_cnn_decoder.png', show_shapes=True)

# instantiate VAE model
# outputs = decoder(merged_model(encoder(inputs)[2:]))
vae = Model(inputs, outputs, name='vae')


# models = (encoder, decoder)
# data = (x_test, y_test) # ???

# VAE loss = mse_loss or xent_loss + kl_loss
# if args.mse:
#     reconstruction_loss = mse(K.flatten(input_fifth),
#                           K.flatten(outputs))
# else:
#     reconstruction_loss = binary_crossentropy(K.flatten(input_fifth),
#                                              K.flatten(outputs))
#
# reconstruction_loss *= image_size[0] * image_size[1] * image_size[2] # ???
# kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
# kl_loss = K.sum(kl_loss, axis=-1)
# kl_loss *= -0.5
# vae_loss = K.mean(reconstruction_loss + kl_loss)
# vae.add_loss(vae_loss)
vae.compile(optimizer='adam', loss=binary_crossentropy)
vae.summary()
plot_model(vae, to_file='vae_cnn.png', show_shapes=True)

from tensorflow.keras.callbacks import TensorBoard
import datetime
currentTimeStr = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
log_dir = os.path.join(
    "logs",
    "fit",
    currentTimeStr,
)
weight_path = 'model/weight.ckpt'
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
from tensorflow.keras.callbacks import ModelCheckpoint
cpCallBack = ModelCheckpoint(weight_dir,
                             monitor='val_loss',
                             verbose=0,
                             save_best_only=True,
                             save_weights_only=True,
                             mode='min',
                             save_freq='epoch')
from tensorflow.keras.callbacks import EarlyStopping
# esCallBack = EarlyStopping(monitor='val_acc',
#                            min_delta=0,
#                            patience=6,
#                            verbose=0,
#                            mode='auto',
#                            baseline=None,
#                            restore_best_weights=False) # ???
from tensorflow.keras.models import load_model
if args.weights:
    vae.load_weights(args.weights)
    vae = load_model('vae_cnn_model.tf')
    vae.fit(x_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_test, y_test),
            callbacks=[tbCallBack, cpCallBack])
    vae.save_weights('vae_cnn_duckie.tf')
    vae.save('vae_cnn_model.tf')
else:
    # train the autoencoder
    vae.fit(x_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_test, y_test),
            callbacks=[tbCallBack, cpCallBack])
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
