from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from keras.layers import Dense, Input, concatenate
from keras.layers import Conv2D, Flatten, Lambda
from keras.layers import Reshape, Conv2DTranspose
from keras.models import Model
from keras.datasets import mnist
from keras.losses import mse, binary_crossentropy
from keras.utils import plot_model
from keras import backend as K

import math
import numpy as np
import matplotlib.pyplot as plt
import argparse
import os


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


def plot_results(models,
                 data,
                 batch_size=128,
                 model_name="vae_mnist"):
    """Plots labels and MNIST digits as function of 2-dim latent vector

    # Arguments
        models (tuple): encoder and decoder models
        data (tuple): test data and label
        batch_size (int): prediction batch size
        model_name (string): which model is using this function
    """

    encoder, decoder = models
    x_test, y_test = data
    os.makedirs(model_name, exist_ok=True)

    filename = os.path.join(model_name, "vae_mean.png")
    # display a 2D plot of the digit classes in the latent space
    z_mean, _, _ = encoder.predict(x_test,
                                   batch_size=batch_size)
    plt.figure(figsize=(12, 10))
    plt.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test)
    plt.colorbar()
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.savefig(filename)
    plt.show()

    filename = os.path.join(model_name, "digits_over_latent.png")
    # display a 30x30 2D manifold of digits
    n = 30
    digit_size = 28
    figure = np.zeros((digit_size * n, digit_size * n))
    # linearly spaced coordinates corresponding to the 2D plot
    # of digit classes in the latent space
    grid_x = np.linspace(-4, 4, n)
    grid_y = np.linspace(-4, 4, n)[::-1]

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample)
            digit = x_decoded[0].reshape(digit_size, digit_size)
            figure[i * digit_size: (i + 1) * digit_size,
                   j * digit_size: (j + 1) * digit_size] = digit

    plt.figure(figsize=(10, 10))
    start_range = digit_size // 2
    end_range = n * digit_size + start_range + 1
    pixel_range = np.arange(start_range, end_range, digit_size)
    sample_range_x = np.round(grid_x, 1)
    sample_range_y = np.round(grid_y, 1)
    plt.xticks(pixel_range, sample_range_x)
    plt.yticks(pixel_range, sample_range_y)
    plt.xlabel("z[0]")
    plt.ylabel("z[1]")
    plt.imshow(figure, cmap='Greys_r')
    plt.savefig(filename)
    plt.show()


# DUCKIETOWN dataset
dataset_file = np.load('duckie_img.npz')
dataset = dataset_file["arr_0"]

x_train, x_test = np.split(dataset, [math.floor(0.7 * dataset.shape[0])])
y_train, y_test = x_train, x_test
print(x_train.shape)
image_size = [x_train.shape[1], x_train.shape[2]]
x_train = np.reshape(x_train, [-1, image_size[0], image_size[1], 1])
x_test = np.reshape(x_test, [-1, image_size[0], image_size[1], 1])
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# network parameters
input_shape = (image_size[0], image_size[1], 1)
batch_size = 36
kernel_size = 5
filters = 16
latent_dim = 50
epochs = 80
strides = (2, 2)

# VAE model = encoder + decoder
# build encoder model
inputs = Input(shape=input_shape, name='encoder_input')
input_four = Lambda(lambda x:
                    x[:, 0:480*4,
                        :, :])(inputs)
print(inputs)
print(input_four)
input_fifth = Lambda(lambda x:
                     x[:, 1920:2400,
                         :, :])(inputs)
print(input_fifth)
actions = Lambda(lambda x:
                    x[:, 480*5, 0:2, 0])(inputs)
x = input_four
for i in range(3):
    x = Conv2D(filters=filters,
               kernel_size=kernel_size,
               activation='relu',
               strides=strides,
               padding='same')(x)
    filters *= 2

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

# use reparameterization trick to push the sampling out as input
# note that "output_shape" isn't necessary with the TensorFlow backend
z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

# instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z, actions], name='encoder')
encoder.summary()
plot_model(encoder, to_file='vae_cnn_encoder.png', show_shapes=True)

# Concatenate
first_input = Input(shape=(latent_dim,))
second_input = Input(shape=(2,))
merged = concatenate([first_input, second_input])
print(merged)
merged_model = Model([first_input, second_input], merged, name='merged_model')

# build decoder model
latent_inputs = Input(shape=(latent_dim + 2,), name='z_sampling')
x = Dense(shape[1] // 4 * shape[2] * shape[3], activation='relu')(latent_inputs)
x = Reshape((shape[1] // 4, shape[2], shape[3]))(x)

for i in range(3):
    filters //= 2
    x = Conv2DTranspose(filters=filters,
                        kernel_size=kernel_size,
                        activation='relu',
                        strides=strides,
                        padding='same')(x)

outputs = Conv2DTranspose(filters=1,
                          kernel_size=kernel_size,
                          activation='sigmoid',
                          padding='same',
                          name='decoder_output')(x)

# instantiate decoder model
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()
plot_model(decoder, to_file='vae_cnn_decoder.png', show_shapes=True)

# instantiate VAE model
print(encoder(inputs)[2])
print(encoder(inputs)[3])
outputs = decoder(merged_model([encoder(inputs)[2], encoder(inputs)[3]]))
vae = Model(inputs, outputs, name='vae')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    help_ = "Load h5 model trained weights"
    parser.add_argument("-w", "--weights", help=help_)
    help_ = "Use mse loss instead of binary cross entropy (default)"
    parser.add_argument("-m", "--mse", help=help_, action='store_true')
    args = parser.parse_args()
    models = (encoder, decoder)
    data = (x_test, y_test)

    # VAE loss = mse_loss or xent_loss + kl_loss
    #if args.mse:
    reconstruction_loss = mse(K.flatten(input_fifth),
                              K.flatten(outputs))
    #else:
    #    reconstruction_loss = binary_crossentropy(K.flatten(inputs),
    #                                              K.flatten(outputs))

    reconstruction_loss *= image_size[0] * image_size[1]
    kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
    kl_loss = K.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = K.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer='adam')
    vae.summary()
    plot_model(vae, to_file='vae_cnn.png', show_shapes=True)

    from keras.callbacks import TensorBoard
    import datetime
    log_dir = os.path.join(
        "logs",
        "fit",
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
    )
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
    from keras.callbacks import ModelCheckpoint
    cpCallBack = ModelCheckpoint("weights.hdf5",
                                 monitor='val_loss',
                                 verbose=0,
                                 save_best_only=True,
                                 save_weights_only=False,
                                 mode='min',
                                 period=1)
    from keras.callbacks import EarlyStopping
    esCallBack = EarlyStopping(monitor='val_loss',
                               min_delta=0,
                               patience=6,
                               verbose=0,
                               mode='auto',
                               baseline=None,
                               restore_best_weights=False)
    from keras.models import load_model
    if args.weights:
        # vae.load_weights(args.weights)
        vae = load_model('vae_cnn_model.h5')
        vae.fit(x_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(x_test, None),
                callbacks=[tbCallBack, cpCallBack, esCallBack])
        vae.save_weights('vae_cnn_duckie.h5')
        vae.save('vae_cnn_model.h5')
    else:
        # train the autoencoder
        vae.fit(x_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(x_test, None),
                callbacks=[tbCallBack, cpCallBack, esCallBack])
        vae.save_weights('vae_cnn_duckie.h5')
        vae.save('vae_cnn_model.h5')

    # plot_results(models, data, batch_size=batch_size, model_name="vae_cnn")
    outputImg = vae.predict(x_test, batch_size=batch_size)
    figure = plt.figure()
    _width = 480 * 5 + 1
    ax = figure.add_subplot(221)
    ax.imshow(np.reshape(x_test[0], [_width, 640]), cmap=plt.cm.gray)
    ax = figure.add_subplot(222)
    ax.imshow(np.reshape(outputImg[0], [480, 640]), cmap=plt.cm.gray)
    ax = figure.add_subplot(223)
    ax.imshow(np.reshape(x_train[0], [_width, 640]), cmap=plt.cm.gray)
    ax = figure.add_subplot(224)
    ax.imshow(np.reshape(vae.predict(x_train, batch_size=batch_size)[0], [480, 640]),
              cmap=plt.cm.gray)
    plt.show()
    # figure = plt.figure()
    # plt.imshow(figure, cmap='Greys_r')
