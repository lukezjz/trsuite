import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import os


gpus = tf.config.experimental.list_physical_devices(device_type='GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


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
        theta = tf.nn.softmax(conv_theta)
        phi = tf.nn.softmax(conv_phi)
        dist = tf.nn.softmax(conv_dist)
        omega = tf.nn.softmax(conv_omega)
        return theta, phi, dist, omega   # shape: (length, length, 25), (length, length, 13), (length, length, 37), (length, length, 25)


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
        outputs = {"theta": [], "phi": [], "dist": [], "omega": []}
        for model in self.models:
            pt, pp, pd, po = model.predict(inputs)
            outputs["theta"].append(pt[0])
            outputs["phi"].append(pp[0])
            outputs["dist"].append(pd[0])
            outputs["omega"].append(po[0])
        for key in outputs:
            outputs[key] = tf.reduce_mean(outputs[key], axis=0)
        return outputs


class GetFeatures:
    def __init__(self, model_directory, length, msa2input_features_cutoff=0.8):
        self.models = []
        for model_weights in os.listdir(model_directory):
            model = TrNet(n_layers=61, n_filters=64, kernel_size=3, dropout_rate=0.0, n_filters_theta=25, n_filters_phi=13, n_filters_dist=37, n_filters_omega=25)
            model.build(input_shape=(1, length, length, 526))
            path = os.path.join(model_directory, model_weights)
            model.load_weights(path)
            print(f"{path} loaded.")
            self.models.append(model)
        self.msa2input_features = MSA2InputFeatures(msa2input_features_cutoff)

    def predict(self, msa):
        input_features = tf.expand_dims(self.msa2input_features.transform(msa), axis=0)
        i = np.random.randint(len(self.models))
        pt, pp, pd, po = self.models[i].predict(input_features)
        output_features = {"theta": pt[0], "phi": pp[0], "dist": pd[0], "omega": po[0]}
        return output_features

    def predict_(self, msa):
        input_features = tf.expand_dims(self.msa2input_features.transform(msa), axis=0)
        output_features = {"theta": [], "phi": [], "dist": [], "omega": []}
        for model in self.models:
            pt, pp, pd, po = model.predict(input_features)
            output_features["theta"].append(pt[0])
            output_features["phi"].append(pp[0])
            output_features["dist"].append(pd[0])
            output_features["omega"].append(po[0])
        for key in output_features:
            output_features[key] = tf.reduce_mean(output_features[key], axis=0)
        return output_features


# def PSSM2Features(pssm_initializer, sample=False):
#     # ref: https://blog.evjang.com/2016/11/tutorial-categorical-variational.html
#     if sample:
#         pssm_ = tf.Variable(pssm_initializer)
#         U = tf.random.uniform(tf.shape(pssm_), minval=0, maxval=1)
#         pssm = tf.nn.softmax(pssm_ - tf.math.log(-tf.math.log(U + 1e-9) + 1e-9))
#     pssm = tf.nn.softmax(pssm_initializer, -1)
#     y_seq = tf.one_hot(tf.argmax(pssm, -1), 20)  # shape: (1, n_seq, L) -> (1, n_seq, L, 20)
#     y_seq = tf.stogradient(y_seq - pssm) + pssm  # gradient bypass


class MSA2InputFeatures:
    def __init__(self, cutoff=0.8, penalty=4.5, diag=0.4):
        self.cutoff = cutoff
        self.penalty = penalty
        self.diag = diag

    def transform(self, msa, pssm=None):
        msa1hot = tf.one_hot(msa, 21, dtype=tf.float32)  # shape: (n_seq, length, 21)
        n_seq, length = msa1hot.shape[:2]

        x_i = msa1hot[0, :, :20]  # shape: (length, 20)

        if pssm:
            f_i = pssm

            h_i = tf.reduce_sum(-f_i * tf.math.log(f_i + 1e-9), axis=1)  # shape: (length, )
            feature1d = tf.concat([x_i, f_i, h_i[:, None]], axis=-1)  # shape: (length, 20+21+1)
            feature1d = tf.reshape(feature1d, [length, 42])

            if n_seq == 1:
                print("not understood!")
                ic = self.diag * tf.eye(length * 21)  # ???
                ic = tf.reshape(ic, (length, 21, length, 21))
                ic = tf.transpose(ic, (0, 2, 1, 3))
                ic = tf.reshape(ic, (length, length, 441))
                i0 = tf.zeros([length, length, 1])
                f2d_dca = tf.concat([ic, i0], axis=-1)
            else:
                raise Exception("not realized!")

        else:
            # ? reweight
            identity_min = tf.cast(length, tf.float32) * self.cutoff
            identity_matrix = tf.tensordot(msa1hot, msa1hot, [[1, 2], [1, 2]])  # shape: (n_seq, n_seq)
            identity_mask = identity_matrix > identity_min
            seq_weights = 1.0 / tf.reduce_sum(tf.cast(identity_mask, dtype=tf.float32), axis=-1)  # weights for each sequence, shape: (n_seq, )
            f_i = tf.reduce_sum(seq_weights[:, None, None] * msa1hot, axis=0) / tf.reduce_sum(seq_weights)  # shape: (length, 21)

            h_i = tf.reduce_sum(-f_i * tf.math.log(f_i + 1e-9), axis=1)  # shape: (length, )
            feature1d = tf.concat([x_i, f_i, h_i[:, None]], axis=-1)  # shape: (length, 20+21+1)
            feature1d = tf.reshape(feature1d, [length, 42])

            # ? f2d_dca
            if n_seq == 1:
                f2d_dca = tf.zeros([length, length, 442], tf.float32)
            else:
                # ? cov
                print("only for PSP!")
                x = tf.reshape(msa1hot, (n_seq, length * 21))   # shape: (n_seq, length * 21)
                num_points = tf.reduce_sum(seq_weights) - tf.sqrt(tf.reduce_mean(seq_weights))   # shape: (,)
                mean = tf.reduce_sum(x * seq_weights[:, None], axis=0, keepdims=True) / num_points   # shape: (1, length * 21)
                x = (x - mean) * tf.sqrt(seq_weights[:, None])
                cov = tf.matmul(tf.transpose(x), x) / num_points
                # ? inverse covariance
                cov_reg = cov + tf.eye(length * 21) * self.penalty / tf.sqrt(tf.reduce_sum(seq_weights))
                inv_cov = tf.linalg.inv(cov_reg)
                x1 = tf.reshape(inv_cov, (length, 21, length, 21))
                x2 = tf.transpose(x1, [0, 2, 1, 3])
                features = tf.reshape(x2, (length, length, 441))
                x3 = tf.sqrt(tf.reduce_sum(tf.square(x1[:, :-1, :, :-1]), (1, 3))) * (1 - tf.eye(length))
                apc = tf.reduce_sum(x3, 0, keepdims=True) * tf.reduce_sum(x3, 1, keepdims=True) / tf.reduce_sum(x3)
                contacts = (x3 - apc) * (1 - tf.eye(length))
                f2d_dca = tf.concat([features, contacts[:, :, None]], axis=-1)

        features2d = tf.concat([tf.tile(feature1d[:, None, :], [1, length, 1]), tf.tile(feature1d[None, :, :], [length, 1, 1]), f2d_dca], axis=-1)
        # shape: [(1, length, length, 42), (1, length, length, 42), (1, length, length, 442)] ---> (1, length, length, 526)
        features2d = tf.reshape(features2d, [length, length, 526])

        return features2d
