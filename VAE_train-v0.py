# coding=utf-8
__version__ = 'v0'

from vae_model import VAE
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy
from tensorflow.keras import backend as K
import tensorflow as tf
import datetime
import math
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import argparse
import os

"""
Description:
    Train model simulating Duckietown environment, using VAE-like model.
LIN
2020/04/13
"""

# for the sake of running on hpc, which does not have display device
matplotlib.use('Agg')

parser = argparse.ArgumentParser()
parser.add_argument('-w', '--weights', help='Load trained weights')
parser.add_argument('-m', '--model', help='Load tf model')
parser.add_argument('-e', '--epoch', help='epoch', default=2, type=int)
args = parser.parse_args()

# logs registry
log_dir = 'v4-logs/'
currentTimeStr = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
scalar_log_dir = os.path.join(log_dir, 'scalars/', currentTimeStr)
loss_file_writer = tf.summary.create_file_writer(scalar_log_dir + "/loss")
loss_file_writer.set_as_default()
# tf.summary.trace_on(graph=True)

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
               'reward_loss': 2e-5,
               'done_loss': 10.0,
               'latent_loss': 5e-3}

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
        # _done_loss = BinaryCrossentropy()(predicted_done, vae.done_var)

        total_loss = loss_weight['image_loss'] * _reconstruction_loss + \
                     loss_weight['latent_loss'] * _kl_loss + \
                     loss_weight['reward_loss'] * _reward_loss \
                     # + loss_weight['done_loss'] * _done_loss

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
    # train_done_loss(_done_loss)

    return predicted_img


@tf.function
def test_step(inputs):
    predicted_img, predicted_reward, predicted_done = vae(inputs, training=False)
    _reconstruction_loss = MeanSquaredError()(predicted_img, vae.input_fifth_var)
    _reward_loss = MeanSquaredError()(predicted_reward, vae.reward_var)
    _kl_loss = -0.5 * (K.sum(
        1 - K.square(vae.z_mean_var)
        - K.exp(vae.z_log_var_var)
        + vae.z_log_var_var,
        axis=-1))
    # _done_loss = BinaryCrossentropy()(predicted_done, vae.done_var)
    total_loss = loss_weight['image_loss'] * _reconstruction_loss + \
                 loss_weight['latent_loss'] * _kl_loss + \
                 loss_weight['reward_loss'] * _reward_loss
                 # + \
                 # loss_weight['done_loss'] * _done_loss

    test_loss(total_loss)
    test_reconstruction_loss(_reconstruction_loss)
    test_reward_loss(_reward_loss)
    test_kl_loss(_kl_loss)
    # test_done_loss(_done_loss)

    return predicted_img


# load DUCKIETOWN data set
print('loading data...')
dataset_file = np.load('dataset_vae.npz')
print('data loaded.')
data_set = dataset_file["arr_0"]
del dataset_file
print('taken out data from data set')
print('data set shape: ', data_set.shape)
# shuffle the data set
print('np shuffling...')
np.random.shuffle(data_set)
print('np shuffle completed.')
# data_set = data_set[:10000]
# data pre-processing
# the following operation maps image value to [0, 1], without affecting value of
# other parameters, i.e. actions, reward, etc.
# notice that type of all values is the same, i.e. 'float32'
image_size = data_set.shape[1:]
print(image_size)
input_shape = (image_size[0], image_size[1], image_size[2])
print('pre-processing...')
data_set = data_set.astype('float32') / 255
print('divide ops done.')
data_set[:, image_size[0] - 1, :, :] *= 255
print('partial mul ops done.')
# print(data_set[10, :, :, :])
# input('Press ENTER to continue...')

# split data set into training and validation sets
# notice that there is no definition on y_train or y_val
x_train, x_val = np.split(data_set, [math.floor(TRAIN_VAL_SPLIT_CONSTANT * data_set.shape[0])])
print('x_train''s shape is: ' + str(x_train.shape))

print('shuffling...')
train_ds = tf.data.Dataset.from_tensor_slices(x_train).shuffle(100000).batch(batch_size)
print('shuffle of train ds completed.')
test_ds = tf.data.Dataset.from_tensor_slices(x_val).batch(batch_size)
print('shuffle completed.')

print('model declaring...')
if args.model:
    print('model loading from file...')
    vae = tf.keras.models.load_model(args.model)
else:
    vae = VAE()
print('model declared.')
# vae.summary()
# plot_model(vae, to_file='vae_cnn.png', show_shapes=True)
ret1 = None
ret2 = None
best_test_loss = math.inf
print('start training...')
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
        ret1 = train_step(batch_inputs)

    for batch_inputs in test_ds:
        # print(batch_inputs.shape)
        ret2 = test_step(batch_inputs)

    if epoch == epochs - 1:
        print('save last epoch''s weight...')
        vae.save_weights(os.path.join('v4-model', currentTimeStr, 'weight{}'.format(epoch+1)), save_format='tf')
        # print('save model...')
        # vae.save(os.path.join('v4-model', currentTimeStr, 'model{}'.format(epoch+1)), save_format='tf')

    # save best model
    if test_loss.result() < best_test_loss:
        print('save best model...')
        best_test_loss = test_loss.result()
        vae.save(os.path.join('v4-model', currentTimeStr, 'best_model{}'.format(epoch+1)), save_format='tf')

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
    tf.summary.scalar(name='train/weighted/reconstruction_loss',
                      data=loss_weight['image_loss'] * train_reconstruction_loss.result(), step=epoch)
    tf.summary.scalar(name='train/weighted/reward_loss',
                      data=loss_weight['reward_loss'] * train_reward_loss.result(), step=epoch)
    tf.summary.scalar(name='train/weighted/kl_loss',
                      data=loss_weight['latent_loss'] * train_kl_loss.result(), step=epoch)
    tf.summary.scalar(name='train/weighted/done_loss',
                      data=loss_weight['done_loss'] * train_done_loss.result(), step=epoch)

    tf.summary.scalar(name='test/total_loss', data=test_loss.result(), step=epoch)
    tf.summary.scalar(name='test/reconstruction_loss', data=test_reconstruction_loss.result(), step=epoch)
    tf.summary.scalar(name='test/reward_loss', data=test_reward_loss.result(), step=epoch)
    tf.summary.scalar(name='test/kl_loss', data=test_kl_loss.result(), step=epoch)
    tf.summary.scalar(name='test/done_loss', data=test_done_loss.result(), step=epoch)
    tf.summary.scalar(name='test/weighted/reconstruction_loss',
                      data=loss_weight['image_loss'] * test_reconstruction_loss.result(), step=epoch)
    tf.summary.scalar(name='test/weighted/reward_loss',
                      data=loss_weight['reward_loss'] * test_reward_loss.result(), step=epoch)
    tf.summary.scalar(name='test/weighted/kl_loss',
                      data=loss_weight['latent_loss'] * test_kl_loss.result(), step=epoch)
    tf.summary.scalar(name='test/weighted/done_loss',
                      data=loss_weight['done_loss'] * test_done_loss.result(), step=epoch)

# tf.summary.scalar(name='loss_weight/kl', data=loss_weight['latent_loss'], step=1)
# tf.summary.scalar(name='loss_weight/done', data=loss_weight['done_loss'], step=1)
# tf.summary.scalar(name='loss_weight/reconstruction', data=loss_weight['image_loss'], step=1)
# tf.summary.scalar(name='loss_weight/reward', data=loss_weight['reward_loss'], step=1)

# tf.summary.trace_export(name='trace_graph', step=0)

fig = plt.figure()
for i in range(0, 3):
    for j in range(0, 5):
        ax = fig.add_subplot(3, 5, i*5+j+1)
        ax.axis('off')
        ax.imshow(ret1[i, :image_size[0] - 1, :, 0], cmap=plt.cm.gray)
plt.subplots_adjust(hspace=0.1)
plt.savefig('sampleVAEimg_train.svg', format='svg', dpi=1200)

fig = plt.figure()
for i in range(0, 3):
    for j in range(0, 5):
        ax = fig.add_subplot(3, 5, i*5+j+1)
        ax.axis('off')
        ax.imshow(ret2[i, :image_size[0] - 1, :, 0], cmap=plt.cm.gray)
plt.subplots_adjust(hspace=0.1)
plt.savefig('sampleVAEimg_test.svg', format='svg', dpi=1200)

# new_model = VAE()
# new_model.load_weights(os.path.join('v4-model', currentTimeStr, 'best_weight1'))
# predictions = vae.predict(x_val[:1])
# print(predictions)
# new_predictions = new_model.predict(x_val[:1])
# print(new_predictions)
# np.testing.assert_allclose(predictions, new_predictions, rtol=1e-6, atol=1e-6)
