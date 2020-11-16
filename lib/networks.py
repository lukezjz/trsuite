import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import os


# bkgrd_folder = "bkgrd01"
# n_layers_ = 36
# Conv2d_kernel0 = np.load(f"../models/{bkgrd_folder}/conv2d-kernel.npy")
# Conv2d_bias0 = np.load(f"../models/{bkgrd_folder}/conv2d-bias.npy")
# InstanceNorm_beta0 = np.load(f"../models/{bkgrd_folder}/InstanceNorm-beta.npy")
# InstanceNorm_gamma0 = np.load(f"../models/{bkgrd_folder}/InstanceNorm-gamma.npy")
# Conv2d_kernel = [np.load(f"../models/{bkgrd_folder}/conv2d_{i}-kernel.npy") for i in range(1, n_layers_ * 2 + 6)]
# Conv2d_bias = [np.load(f"../models/{bkgrd_folder}/conv2d_{i}-bias.npy") for i in range(1, n_layers_ * 2 + 6)]
# InstanceNorm_beta = [np.load(f"../models/{bkgrd_folder}/InstanceNorm_{i}-beta.npy") for i in range(1, n_layers_ * 2 + 1)]
# InstanceNorm_gamma = [np.load(f"../models/{bkgrd_folder}/InstanceNorm_{i}-gamma.npy") for i in range(1, n_layers_ * 2 + 1)]


# model_folder = "model01"
# n_layers_ = 61
# Conv2d_kernel0 = np.load(f"../models/{model_folder}/conv2d-kernel.npy")
# Conv2d_bias0 = np.load(f"../models/{model_folder}/conv2d-bias.npy")
# InstanceNorm_beta0 = np.load(f"../models/{model_folder}/InstanceNorm-beta.npy")
# InstanceNorm_gamma0 = np.load(f"../models/{model_folder}/InstanceNorm-gamma.npy")
# Conv2d_kernel = [np.load(f"../models/{model_folder}/conv2d_{i}-kernel.npy") for i in range(1, n_layers_ * 2 + 6)]
# Conv2d_bias = [np.load(f"../models/{model_folder}/conv2d_{i}-bias.npy") for i in range(1, n_layers_ * 2 + 6)]
# InstanceNorm_beta = [np.load(f"../models/{model_folder}/InstanceNorm_{i}-beta.npy") for i in range(1, n_layers_ * 2 + 1)]
# InstanceNorm_gamma = [np.load(f"../models/{model_folder}/InstanceNorm_{i}-gamma.npy") for i in range(1, n_layers_ * 2 + 1)]


class ResNetBlock(tf.keras.layers.Layer):   # tested
    def __init__(self, n_filters, kernel_size, dilation_rate, dropout_rate, initializers=None):
        super(ResNetBlock, self).__init__()
        layers = ['conv1_kernel', 'conv1_bias', 'in1_beta', 'in1_gamma', 'conv2_kernel', 'conv2_bias', 'in2_beta', 'in2_gamma']
        if initializers is None:
            initializers = {}
        for layer in layers:
            if layer not in initializers:
                initializers[layer] = None

        self.identity = lambda x: x
        self.conv1 = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=kernel_size, strides=1, padding="same", dilation_rate=dilation_rate)
        #                                    kernel_initializer=tf.keras.initializers.constant(initializers['conv1_kernel']),
        #                                    bias_initializer=tf.keras.initializers.constant(initializers['conv1_bias']))
        self.in1 = tfa.layers.InstanceNormalization()
        #                                            beta_initializer=tf.keras.initializers.constant(initializers['in1_beta']),
        #                                            gamma_initializer=tf.keras.initializers.constant(initializers['in1_gamma']))
        self.elu1 = tf.keras.layers.Activation(tf.keras.activations.elu)
        self.dropout = tf.keras.layers.Dropout(rate=dropout_rate)
        self.conv2 = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=kernel_size, strides=1, padding="same", dilation_rate=dilation_rate)
        #                                    kernel_initializer=tf.keras.initializers.constant(initializers['conv2_kernel']),
        #                                    bias_initializer=tf.keras.initializers.constant(initializers['conv2_bias']))
        self.in2 = tfa.layers.InstanceNormalization()
        #                                            beta_initializer=tf.keras.initializers.constant(initializers['in2_beta']),
        #                                            gamma_initializer=tf.keras.initializers.constant(initializers['in2_gamma']))
        self.elu2 = tf.keras.layers.Activation(tf.keras.activations.elu)

    def call(self, inputs, **kwargs):
        identity = self.identity(inputs)
        conv1 = self.conv1(inputs)
        in1 = self.in1(conv1)
        elu1 = self.elu1(in1)
        dropout = self.dropout(elu1, training=False)  # ?
        conv2 = self.conv2(dropout)
        in2 = self.in2(conv2)
        output = self.elu2(tf.keras.layers.add([identity, in2]))
        return output


class TrNet(tf.keras.Model):   # tested
    def __init__(self, n_layers, n_filters, kernel_size, dropout_rate, n_filters_theta, n_filters_phi, n_filters_dist, n_filters_omega):
        super(TrNet, self).__init__()
        self.n_layers = n_layers
        self.n_filters = n_filters
        self.conv = tf.keras.layers.Conv2D(filters=n_filters, kernel_size=1, strides=1, padding="same")
        #                                   kernel_initializer=tf.keras.initializers.constant(Conv2d_kernel0), bias_initializer=tf.keras.initializers.constant(Conv2d_bias0))
        self.in_ = tfa.layers.InstanceNormalization()
        #                                            beta_initializer=tf.keras.initializers.constant(InstanceNorm_beta0),
        #                                            gamma_initializer=tf.keras.initializers.constant(InstanceNorm_gamma0))
        self.elu = tf.keras.layers.Activation(tf.keras.activations.elu)
        dilation_rates = [1, 2, 4, 8, 16]
        # initializers_ = [{'conv1_kernel': Conv2d_kernel[2 * j], 'conv1_bias': Conv2d_bias[2 * j],
        #                   'in1_beta': InstanceNorm_beta[2 * j], 'in1_gamma': InstanceNorm_gamma[2 * j],
        #                   'conv2_kernel': Conv2d_kernel[2 * j + 1], 'conv2_bias': Conv2d_bias[2 * j + 1],
        #                   'in2_beta': InstanceNorm_beta[2 * j + 1], 'in2_gamma': InstanceNorm_gamma[2 * j + 1]} for j in range(n_layers)]
        ## hybrid dilated convolution (HDC)
        self.resnet_blocks = [ResNetBlock(n_filters=n_filters, kernel_size=kernel_size, dilation_rate=dilation_rates[i % 5], dropout_rate=dropout_rate) for i in range(n_layers)]
        #                                  initializers=initializers_[i]) for i in range(n_layers)]
        self.conv_theta = tf.keras.layers.Conv2D(filters=n_filters_theta, kernel_size=1, strides=1, padding="same")
        #                                         kernel_initializer=tf.keras.initializers.constant(Conv2d_kernel[-5]),
        #                                         bias_initializer=tf.keras.initializers.constant(Conv2d_bias[-5]))
        self.conv_phi = tf.keras.layers.Conv2D(filters=n_filters_phi, kernel_size=1, strides=1, padding="same")
        #                                       kernel_initializer=tf.keras.initializers.constant(Conv2d_kernel[-4]),
        #                                       bias_initializer=tf.keras.initializers.constant(Conv2d_bias[-4]))
        self.conv_dist = tf.keras.layers.Conv2D(filters=n_filters_dist, kernel_size=1, strides=1, padding="same")
        #                                        kernel_initializer=tf.keras.initializers.constant(Conv2d_kernel[-3]),
        #                                        bias_initializer=tf.keras.initializers.constant(Conv2d_bias[-3]))
        self.conv_omega = tf.keras.layers.Conv2D(filters=n_filters_omega, kernel_size=1, strides=1, padding="same")
        #                                         kernel_initializer=tf.keras.initializers.constant(Conv2d_kernel[-1]),
        #                                         bias_initializer=tf.keras.initializers.constant(Conv2d_bias[-1]))

    def call(self, inputs, **kwargs):
        conv = self.conv(inputs)
        in_ = self.in_(conv)
        h = self.elu(in_)
        for i in range(self.n_layers):
            h = self.resnet_blocks[i](h)
        conv_theta, conv_phi = self.conv_theta(h), self.conv_phi(h)
        symmetry = 0.5 * (h + tf.transpose(h, perm=[0, 2, 1, 3]))
        conv_dist, conv_omega = self.conv_dist(symmetry), self.conv_omega(symmetry)
        p_theta = tf.nn.softmax(conv_theta)
        p_phi = tf.nn.softmax(conv_phi)
        p_dist = tf.nn.softmax(conv_dist)
        p_omega = tf.nn.softmax(conv_omega)
        return p_theta, p_phi, p_dist, p_omega   # shape: (length, length, 25), (length, length, 13), (length, length, 37), (length, length, 25)


class GetBackground:
    def __init__(self, background_directory, length):
        self.models = []
        self.length = length
        for bkgrd_weights in os.listdir(background_directory):
            model = TrNet(n_layers=36, n_filters=64, kernel_size=3, dropout_rate=0.15, n_filters_theta=25, n_filters_phi=13, n_filters_dist=37, n_filters_omega=25)
            model.build(input_shape=(1, length, length, 64))
            path = os.path.join(background_directory, bkgrd_weights)
            model.load_weights(path)
            print(f"{path} loaded.")
            self.models.append(model)

    def generate(self, seed=None):
        # self.models[0].summary()
        if seed is not None:
            np.random.seed(seed)
        inputs = np.random.normal(size=(5, self.length, self.length, 64))
        outputs = {"p_theta": [], "p_phi": [], "p_dist": [], "p_omega": []}
        for model in self.models:
            pt, pp, pd, po = model.predict(inputs)
            outputs["p_theta"].append(pt[0])
            outputs["p_phi"].append(pp[0])
            outputs["p_dist"].append(pd[0])
            outputs["p_omega"].append(po[0])
        for key in outputs:
            outputs[key] = np.mean(outputs[key], axis=0)
        return outputs


class GetFeatures:
    def __init__(self, model_directory, length_, msa2input_features_cutoff=0.8):
        self.models = []
        for model_weights in os.listdir(model_directory):
            model = TrNet(n_layers=61, n_filters=64, kernel_size=3, dropout_rate=0.0, n_filters_theta=25, n_filters_phi=13, n_filters_dist=37, n_filters_omega=25)
            model.build(input_shape=(1, length_, length_, 526))
            path = os.path.join(model_directory, model_weights)
            model.load_weights(path)
            print(f"{path} loaded.")
            self.models.append(model)
        self.msa2input_features = MSA2InputFeatures(msa2input_features_cutoff)

    def predict(self, msa):
        input_features = self.msa2input_features.transform(msa)
        output_features = {"p_theta": [], "p_phi": [], "p_dist": [], "p_omega": []}
        for model in self.models:
            pt, pp, pd, po = model.predict(input_features)
            output_features["p_theta"].append(pt[0])
            output_features["p_phi"].append(pp[0])
            output_features["p_dist"].append(pd[0])
            output_features["p_omega"].append(po[0])

        for key in output_features:
            output_features[key] = np.mean(output_features[key], axis=0)

        return output_features


class MSA2InputFeatures:
    def __init__(self, cutoff=0.8, penalty=4.5):
        self.cutoff = cutoff
        self.penalty = penalty

    def transform(self, msa):
        msa1hot = tf.one_hot(msa, 21, dtype=tf.float32)  # shape: (n_seq, length_, 21)
        n_seq, length_ = msa1hot.shape[:2]

        # ? reweight msa
        id_min = tf.cast(length_, tf.float32) * self.cutoff  # 1d
        id_mtx = tf.tensordot(msa1hot, msa1hot, [[1, 2], [1, 2]])  # shape: (n_seq, n_seq): (1, 1)
        id_mask = id_mtx > id_min
        weights = 1.0 / tf.reduce_sum(tf.cast(id_mask, dtype=tf.float32), axis=-1)  # (n_seq, )

        # ? f1d_pssm
        f1d_seq = msa1hot[0, :, :20]  # shape: (length_, 20)
        f_i = tf.reduce_sum(weights[:, None, None] * msa1hot, axis=0) / tf.reduce_sum(weights) + 1e-9  # shape: (length_, 21)
        h_i = tf.reduce_sum(-f_i * tf.math.log(f_i), axis=1)  # shape: (length_, )
        f1d_pssm = tf.concat([f1d_seq, f_i, h_i[:, None]], axis=1)  # shape: (length_, 20+21+1)
        f1d = tf.expand_dims(f1d_pssm, axis=0)  # shape: (1, length_, 42)
        f1d = tf.reshape(f1d, [1, length_, 42])

        # ? f2d_dca
        if n_seq == 1:
            f2d_dca = tf.zeros([1, length_, length_, 442], tf.float32)
        else:
            # ? cov
            x = tf.reshape(msa1hot, (n_seq, length_ * 21))
            num_points = tf.reduce_sum(weights) - tf.sqrt(tf.reduce_mean(weights))
            mean = tf.reduce_sum(x * weights[:, None], axis=0, keepdims=True) / num_points
            x = (x - mean) * tf.sqrt(weights[:, None])
            cov = tf.matmul(tf.transpose(x), x) / num_points
            # ? inverse covariance
            cov_reg = cov + tf.eye(length_ * 21) * self.penalty / tf.sqrt(tf.reduce_sum(weights))
            inv_cov = tf.linalg.inv(cov_reg)
            x1 = tf.reshape(inv_cov, (length_, 21, length_, 21))
            x2 = tf.transpose(x1, [0, 2, 1, 3])
            features = tf.reshape(x2, (length_, length_, 21 * 21))
            x3 = tf.sqrt(tf.reduce_sum(tf.square(x1[:, :-1, :, :-1]), (1, 3))) * (1 - tf.eye(length_))
            apc = tf.reduce_sum(x3, 0, keepdims=True) * tf.reduce_sum(x3, 1, keepdims=True) / tf.reduce_sum(x3)
            contacts = (x3 - apc) * (1 - tf.eye(length_))
            f2d_dca = tf.concat([features, contacts[:, :, None]], axis=2)[None, :]

        f2d = tf.concat([tf.tile(f1d[:, :, None, :], [1, 1, length_, 1]), tf.tile(f1d[:, None, :, :], [1, length_, 1, 1]), f2d_dca], axis=-1)
        # shape: [(1, length_, length_, 42), (1, length_, length_, 42), (1, length_, length_, 442)] ---> (1, length_, length_, 526)
        features_2d = tf.reshape(f2d, [1, length_, length_, 526])

        return features_2d
