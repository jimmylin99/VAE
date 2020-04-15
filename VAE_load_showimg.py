from vae_model import VAE
import matplotlib.pyplot as plt
import numpy as np
from tensorflow.keras.models import load_model
import math
import os

# vae = VAE()
# vae.load_weights('tmp_weight/weight200')

vae = load_model('v4-model/20200415-213615/best_model120')

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
for i in range(data_set.shape[0]):
    if data_set[i, image_size[0] - 1, 6, 0] < -20:
        data_set[i, image_size[0] - 1, 6, :] = -20
print('clip reward done.')
print('shrinking data set size...')
data_set = data_set[:300]
sample_num = data_set.shape[0]
print('sample num: {}'.format(sample_num))
# print(data_set[10, :, :, :])
# input('Press ENTER to continue...')

row = 4
col = 5
ret = vae.predict(data_set[:row*col])
print(ret)
predicted_img = ret[0]

fig = plt.figure()
for i in range(0, row):
    for j in range(0, col):
        index = i * row + j
        ax = fig.add_subplot(row, col, index + 1)
        ax.axis('off')
        ax.imshow(predicted_img[index, :image_size[0] - 1, :, 0], cmap=plt.cm.gray)
plt.subplots_adjust(hspace=0.1)
plt.savefig('sampleVAEimg_predicted.svg', format='svg', dpi=1200)

fig = plt.figure()
for i in range(0, row):
    for j in range(0, col):
        index = i * row + j
        ax = fig.add_subplot(row, col, index + 1)
        ax.axis('off')
        ax.imshow(data_set[index, :image_size[0] - 1, :, 0], cmap=plt.cm.gray)
plt.subplots_adjust(hspace=0.1)
plt.savefig('sampleVAEimg_original.svg', format='svg', dpi=1200)

ret = vae.predict(data_set)
predicted_reward = ret[1]

print('predicting...')
for i in range(sample_num):
    print(predicted_reward[i, 0], data_set[i, image_size[0] - 1, 6, 0])

x = range(sample_num)
plt.figure()
plt.plot(x, predicted_reward[:, 0], 'r--', label='predicted')
plt.plot(x, data_set[:, image_size[0] - 1, 6, 0], 'b--', label='ground_truth')
plt.ylabel('reward')
plt.legend()
plt.savefig('sample_reward_preview.svg', format='svg', dpi=1200)

filtered_predict_reward = []
filtered_ground_truth_reward = []
for i in range(sample_num):
    if predicted_reward[i, 0] < -20 or data_set[i, image_size[0] - 1, 6, 0] < -20:
        continue
    filtered_predict_reward.append(predicted_reward[i, 0])
    filtered_ground_truth_reward.append(data_set[i, image_size[0] - 1, 6, 0])

Z = zip(filtered_ground_truth_reward, filtered_predict_reward)
Z = sorted(Z, reverse=True)
filtered_ground_truth_reward, filtered_predict_reward = zip(*Z)

x = range(len(filtered_ground_truth_reward))
plt.figure()
plt.plot(x, filtered_predict_reward, 'ro-', label='predicted')
plt.plot(x, filtered_ground_truth_reward, 'bo-', label='ground_truth')
plt.ylabel('reward')
plt.legend()
plt.savefig('sample_filtered_reward_preview.svg', format='svg', dpi=1200)

plt.show()
