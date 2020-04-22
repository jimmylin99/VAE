from tensorflow.keras.layers import Layer, Dense, Concatenate, Conv2D, Flatten, Lambda, Reshape, Conv2DTranspose
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow import shape, exp

"""
    TODO: activation of reconstruction layer, whether appropriate to use softmax
"""


class Sampling(Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def __init__(self,
                 name='sampling'):
        super(Sampling, self).__init__(name=name)

    def call(self, inputs, **kwargs):
        z_mean, z_log_var = inputs
        batch = shape(z_mean)[0]
        dim = shape(z_mean)[1]
        epsilon = K.random_normal(shape=(batch, dim))
        return z_mean + exp(0.5 * z_log_var) * epsilon


class VAE(Model):
    # VAE model = encoder + decoder
    # [input_four + action] --> [predicted_{img + reward + done}]
    def __init__(self,
                 latent_dim=20,
                 image_size=None,
                 name='VAE'):
        super(VAE, self).__init__(name=name)

        if image_size is None:
            image_size = (121, 160, 5)

        # categorize input
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

        self.sampling = Sampling()

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
        self.input_var = inputs
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
        self.predicted_reward_var = K.clip(self.predicted_reward_var, -20.0, 20.0)

        x = self.dd1(merged)
        x = self.dd2(x)
        x = self.dd3(x)
        x = self.dd4(x)
        self.predicted_done_var = self.ddone(x)

        return self.predicted_img_var, self.predicted_reward_var, self.predicted_done_var

