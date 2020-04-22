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
# for the sake of running on hpc, which does not have display device
# also, it has to be used before importing matplotlib.pyplot
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
import os

"""
Description:
    Train model simulating Duckietown environment, using VAE-like model.
    update(w.r.t. v0): Use data from gen v3.
    update: arg -s -r added
LIN
2020/04/22
"""

parser = argparse.ArgumentParser()
parser.add_argument('-w', '--weights', help='Load trained weights')
parser.add_argument('-m', '--model', help='Load tf model')
parser.add_argument('-e', '--epoch', help='epoch', default=2, type=int)
parser.add_argument('-b', '--batch', help='batch size', default=128, type=int)
parser.add_argument('-s', '--earlystop', help='early stop tolerance step', default=20, type=int)
parser.add_argument('-r', '--randomshuffle', help='random shuffle buffer size', default=1000, type=int)
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
batch_size = args.batch
latent_dim = 20
epochs = args.epoch
loss_weight = {'image_loss': 10.0,
               'reward_loss': 0.01,
               'done_loss': 10.0,
               'latent_loss': 1e-3}

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
def train_step_branch(inputs):
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

        # bough_loss = loss_weight['image_loss'] * _reconstruction_loss + \
        #              loss_weight['latent_loss'] * _kl_loss
        branch_loss = _reward_loss
        # total_loss = loss_weight['image_loss'] * _reconstruction_loss + \
        #              loss_weight['latent_loss'] * _kl_loss + \
        #              loss_weight['reward_loss'] * _reward_loss \
        #              # + loss_weight['done_loss'] * _done_loss

    reward_trainable_vairables = vae.dr1.trainable_variables + vae.dr2.trainable_variables + \
                                 vae.dr3.trainable_variables + vae.dr4.trainable_variables + \
                                 vae.dreward.trainable_variables
    gradients_reward = tape.gradient(branch_loss, reward_trainable_vairables)
    optimizer.apply_gradients((zip(gradients_reward, reward_trainable_vairables)))

    train_loss(branch_loss)
    # train_loss(total_loss)
    train_reward_loss(_reward_loss)
    train_reconstruction_loss(_reconstruction_loss)
    train_kl_loss(_kl_loss)

    # return tape, bough_loss, branch_loss


@tf.function
def train_step_bough(inputs):
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

        bough_loss = loss_weight['image_loss'] * _reconstruction_loss + \
                     loss_weight['latent_loss'] * _kl_loss
        # branch_loss = _reward_loss
        # total_loss = loss_weight['image_loss'] * _reconstruction_loss + \
        #              loss_weight['latent_loss'] * _kl_loss + \
        #              loss_weight['reward_loss'] * _reward_loss \
        #              # + loss_weight['done_loss'] * _done_loss

    gradients = tape.gradient(bough_loss, vae.trainable_variables)
    optimizer.apply_gradients(zip(gradients, vae.trainable_variables))

    train_loss(bough_loss)
    # train_loss(total_loss)
    train_reward_loss(_reward_loss)
    train_reconstruction_loss(_reconstruction_loss)
    train_kl_loss(_kl_loss)


def train_step(inputs, train_branch=False):
    if train_branch:
        train_step_branch(inputs)
    else:
        train_step_bough(inputs)

    # if train_branch:
    #     reward_trainable_vairables = vae.dr1.trainable_variables + vae.dr2.trainable_variables + \
    #                                  vae.dr3.trainable_variables + vae.dr4.trainable_variables + \
    #                                  vae.dreward.trainable_variables
    #     gradients_reward = tape.gradient(branch_loss, reward_trainable_vairables)
    #     optimizer.apply_gradients((zip(gradients_reward, reward_trainable_vairables)))
    # else:
    #     gradients = tape.gradient(bough_loss, vae.trainable_variables)
    #     optimizer.apply_gradients(zip(gradients, vae.trainable_variables))

    # reward_trainable_vairables = vae.dr1.trainable_variables + vae.dr2.trainable_variables + \
    #                              vae.dr3.trainable_variables + vae.dr4.trainable_variables + \
    #                              vae.dreward.trainable_variables
    # print(reward_trainable_vairables)
    # gradients_reward = tape.gradient(_reward_loss, reward_trainable_vairables)
    # optimizer.apply_gradients((zip(gradients_reward, reward_trainable_vairables)))

    # if train_branch:
    #     train_loss(branch_loss)
    # else:
    #     train_loss(bough_loss)
    # # train_loss(total_loss)
    # train_reward_loss(_reward_loss)
    # train_reconstruction_loss(_reconstruction_loss)
    # train_kl_loss(_kl_loss)
    # train_done_loss(_done_loss)

    # return predicted_img


@tf.function
def test_step(inputs, train_branch=False):
    predicted_img, predicted_reward, predicted_done = vae(inputs, training=False)
    _reconstruction_loss = MeanSquaredError()(predicted_img, vae.input_fifth_var)
    _reward_loss = MeanSquaredError()(predicted_reward, vae.reward_var)
    _kl_loss = -0.5 * (K.sum(
        1 - K.square(vae.z_mean_var)
        - K.exp(vae.z_log_var_var)
        + vae.z_log_var_var,
        axis=-1))
    # _done_loss = BinaryCrossentropy()(predicted_done, vae.done_var)
    # total_loss = loss_weight['image_loss'] * _reconstruction_loss + \
    #              loss_weight['latent_loss'] * _kl_loss + \
    #              loss_weight['reward_loss'] * _reward_loss
    #              # + \
    #              # loss_weight['done_loss'] * _done_loss
    bough_loss = loss_weight['image_loss'] * _reconstruction_loss + \
                 loss_weight['latent_loss'] * _kl_loss
    branch_loss = _reward_loss

    if train_branch:
        test_loss(branch_loss)
    else:
        test_loss(bough_loss)
    # test_loss(total_loss)
    test_reconstruction_loss(_reconstruction_loss)
    test_reward_loss(_reward_loss)
    test_kl_loss(_kl_loss)
    # test_done_loss(_done_loss)

    # return predicted_img


# load DUCKIETOWN data set
print('loading data...')
data_set = np.load('dataset_vae.npy')
print('data loaded.')
# data_set = dataset_file["arr_0"]
# del dataset_file
# print('taken out data from data set')
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
# print('pre-processing...')
# data_set = data_set.astype('float32') / 255
# print('divide ops done.')
# data_set[:, image_size[0] - 1, :, :] *= 255
# print('partial mul ops done.')
# for i in range(data_set.shape[0]):
#     if data_set[i, image_size[0] - 1, 6, 0] < -10:
#         data_set[i, image_size[0] - 1, 6, :] = -10
# print('clip reward done.')
# print(data_set[10, :, :, :])
# input('Press ENTER to continue...')

# split data set into training and validation sets
# notice that there is no definition on y_train or y_val
x_train, x_val = np.split(data_set, [math.floor(TRAIN_VAL_SPLIT_CONSTANT * data_set.shape[0])])
print('x_train''s shape is: ' + str(x_train.shape))

print('shuffling...')
train_ds = tf.data.Dataset.from_tensor_slices(x_train).shuffle(args.randomshuffle).batch(batch_size)
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
# ret1 = None
# ret2 = None
best_test_loss = math.inf
last_best_epoch = 0
best_model_route = None
train_branch = False
automatic_switch = True
print('hyper-parameter for weight: img:{}, reward:{}, kl:{}, done:{}'.format(loss_weight['image_loss'],
                                                                             loss_weight['reward_loss'],
                                                                             loss_weight['latent_loss'],
                                                                             loss_weight['done_loss']))
print('start training...')
for epoch in range(epochs):
    if automatic_switch:
        # automatic branch switching and early stop
        if epoch - last_best_epoch > args.earlystop:
            if train_branch:
                # early stop
                print('early stop before epoch {}'.format(epoch+1))
                break
            else:
                # switch to train branch
                train_branch = True
                last_best_epoch = epoch
                best_test_loss = math.inf
    else:
        # manual switch
        if epoch < 2:
            train_branch = False
        elif train_branch is False:
            # switch to train branch
            train_branch = True
            last_best_epoch = epoch
            best_test_loss = math.inf

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

    print('start training for epoch {}'.format(epoch+1))
    for batch_inputs in train_ds:
        # print(batch_inputs.shape)
        train_step(batch_inputs, train_branch=train_branch)

    print('start validation for epoch {}'.format(epoch+1))
    for batch_inputs in test_ds:
        # print(batch_inputs.shape)
        test_step(batch_inputs, train_branch=train_branch)

    if epoch == epochs - 1:
        print('save last epoch''s weight...')
        vae.save_weights(os.path.join('v4-model', currentTimeStr, 'weight{}'.format(epoch+1)), save_format='tf')
        # print('save model...')
        # vae.save(os.path.join('v4-model', currentTimeStr, 'model{}'.format(epoch+1)), save_format='tf')

    # update best test loss
    if test_loss.result() < best_test_loss:
        last_best_epoch = epoch
        best_test_loss = test_loss.result()
        # save best model only when training branch
        if train_branch:
            print('save best model...')
            _route = os.path.join('v4-model', currentTimeStr, 'best_model{}'.format(epoch+1))
            vae.save(_route, save_format='tf')
            best_model_route = _route
        else:
            print('save bough best model...')
            _route = os.path.join('v4-model', currentTimeStr, 'bough_best_model{}'.format((epoch+1)))
            vae.save(_route, save_format='tf')
            best_model_route = _route

    template = 'Epoch {}, Loss: {}, reconstruction loss: {}, reward loss: {}, kl loss: {}, done loss: {}\n' \
               '----> Test Loss: {}, reconstruction loss: {}, reward loss: {}, kl loss: {}, done loss: {}'
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

tf.summary.scalar(name='loss_weight/kl', data=loss_weight['latent_loss'], step=0)
tf.summary.scalar(name='loss_weight/done', data=loss_weight['done_loss'], step=0)
tf.summary.scalar(name='loss_weight/reconstruction', data=loss_weight['image_loss'], step=0)
tf.summary.scalar(name='loss_weight/reward', data=loss_weight['reward_loss'], step=0)

# tf.summary.trace_export(name='trace_graph', step=0)

# sample_num = 300
# test_samples = x_val[:sample_num]
# predicted_test_samples = vae.predict(test_samples)
# predicted_test_img = predicted_test_samples[0]
# predicted_test_reward = predicted_test_samples[1]
#
# row = 4
# col = 5
#
# fig = plt.figure()
# for i in range(0, row):
#     for j in range(0, col):
#         index = i * row + j
#         ax = fig.add_subplot(row, col, index + 1)
#         ax.axis('off')
#         ax.imshow(predicted_test_img[index, :image_size[0] - 1, :, 0], cmap=plt.cm.gray)
# plt.subplots_adjust(hspace=0.1)
# plt.savefig('sampleVAEimg_predicted.svg', format='svg', dpi=1200)
#
# fig = plt.figure()
# for i in range(0, row):
#     for j in range(0, col):
#         index = i * row + j
#         ax = fig.add_subplot(row, col, index + 1)
#         ax.axis('off')
#         ax.imshow(x_val[index, :image_size[0] - 1, :, 0], cmap=plt.cm.gray)
# plt.subplots_adjust(hspace=0.1)
# plt.savefig('sampleVAEimg_original.svg', format='svg', dpi=1200)
#
# filtered_predict_reward = []
# filtered_ground_truth_reward = []
# for i in range(sample_num):
#     if predicted_test_reward[i, 0] < -20 or data_set[i, image_size[0] - 1, 6, 0] < -20:
#         continue
#     filtered_predict_reward.append(predicted_test_reward[i, 0])
#     filtered_ground_truth_reward.append(data_set[i, image_size[0] - 1, 6, 0])
#
# Z = zip(filtered_ground_truth_reward, filtered_predict_reward)
# Z = sorted(Z, reverse=True)
# filtered_ground_truth_reward, filtered_predict_reward = zip(*Z)
#
# x = range(len(filtered_ground_truth_reward))
# plt.figure()
# plt.plot(x, filtered_predict_reward, 'ro-', label='predicted')
# plt.plot(x, filtered_ground_truth_reward, 'bo-', label='ground_truth')
# plt.ylabel('reward')
# plt.legend()
# plt.savefig('sample_filtered_reward_preview.svg', format='svg', dpi=1200)
#
# plt.show()


# print('show sample img...')
# fig = plt.figure()
# for i in range(0, 3):
#     for j in range(0, 5):
#         index = i*5+j
#         ax = fig.add_subplot(3, 5, index + 1)
#         ax.axis('off')
#         ax.imshow(ret1[index, :image_size[0] - 1, :, 0], cmap=plt.cm.gray)
# plt.subplots_adjust(hspace=0.1)
# plt.savefig('sampleVAEimg_train.svg', format='svg', dpi=1200)
#
# fig = plt.figure()
# for i in range(0, 3):
#     for j in range(0, 5):
#         index = i*5+j
#         ax = fig.add_subplot(3, 5, index + 1)
#         ax.axis('off')
#         ax.imshow(ret2[index, :image_size[0] - 1, :, 0], cmap=plt.cm.gray)
# plt.subplots_adjust(hspace=0.1)
# plt.savefig('sampleVAEimg_test.svg', format='svg', dpi=1200)
# print('img plot done.')

# new_model = VAE()
# new_model.load_weights(os.path.join('v4-model', currentTimeStr, 'best_weight1'))
# predictions = vae.predict(x_val[:1])
# print(predictions)
# new_predictions = new_model.predict(x_val[:1])
# print(new_predictions)
# np.testing.assert_allclose(predictions, new_predictions, rtol=1e-6, atol=1e-6)

print('finished.')
