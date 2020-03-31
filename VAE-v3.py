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
x_train, x_test = np.split(dataset, [math.floor(0.8 * dataset.shape[0])])
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
latent_dim = 15
epochs = args.epoch

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
"""
    Conv2D layer: Conv2D(filter_num, filter_size, activation, strides, padding)
    padding = 'valid': H = ceil((H1 - filter_Height + 1) / stride)
    padding = 'same':  H = ceil(H1 / stride)
"""
x = input_four  # (?, 120, 160, 4)
x = Conv2D(12, (5, 5), activation='relu', strides=(2, 2), padding='valid')(x)  # (?, 58, 78, 12)
x = Conv2D(24, (5, 5), activation='relu', strides=(2, 2), padding='valid')(x)  # (?, 27, 37, 24)
x = Conv2D(36, (5, 5), activation='relu', strides=(2, 2), padding='valid')(x)  # (?, 12, 17, 36)
x = Conv2D(48, (5, 5), activation='relu', strides=(1, 1), padding='valid')(x)  # (?, 8, 13, 48)
x = Conv2D(48, (3, 3), activation='relu', strides=(1, 1), padding='valid')(x)  # (?, 6, 11, 48)
x = Conv2D(64, (3, 3), activation='relu', strides=(1, 1), padding='valid')(x)  # (?, 4, 9, 64)

# shape info needed to build decoder model
shape = K.int_shape(x)

x = Flatten()(x)
x = Dense(1000, activation='relu')(x)  # (?, 1000)
x = Dense(100, activation='relu')(x)  # (?, 100)


def clip_activation(x_min, x_max):
    def wrapped_activation(x):
        return K.clip(x, x_min, x_max)
    return wrapped_activation


z_mean = Dense(latent_dim, activation='linear', name='z_mean')(x)
z_log_var = Dense(latent_dim, activation=clip_activation(-100.0, 10.0), name='z_log_var')(x)

z = Lambda(sampling, output_shape=(latent_dim,), name='z')([z_mean, z_log_var])

merged = concatenate([z, actions, speed])

# build decoder model
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
predicted_img = Conv2DTranspose(1, (6, 6),
                                activation=clip_activation(x_min=0.0, x_max=1.0),
                                strides=(2, 2),
                                padding='valid')(x)   # (?, 120, 160, 1)
# predicted_img = Flatten()(predicted_img)  # (?, 120 * 160 * 1) ??necessary?

x = Dense(100, activation='relu')(merged)
x = Dense(1000, activation='relu')(x)
x = Dense(1000, activation='relu')(x)
x = Dense(100, activation='relu')(x)
predicted_speed = Dense(1, activation='relu')(x)  # (?, 1)

outputs = [predicted_img, predicted_speed]

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


def custom_loss(y_true, y_pred):
    pass


vae.compile(optimizer='adam', loss=custom_loss)
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
