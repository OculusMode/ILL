import numpy as np
from numpy import random, linalg as LA
# import matplotlib.pyplot as plt
from keras.layers import Dense, Layer, Input, Concatenate, Reshape
from keras import Model
import os
# %tensorflow_version 2.x
import tensorflow as tf

device_name = tf.test.gpu_device_name()
if device_name != "/device:GPU:0":
    device_name = "/cpu:0"
else:
    device_name = "/gpu:0"

with tf.device(device_name):

    def avg_dist(y_true, y_pred):
        return tf.reduce_sum(tf.abs(y_true - y_pred), axis=1)

    def avg_norm(y_true, y_pred):
        return tf.norm(y_true - y_pred, axis=1)

    def success(y_true, y_pred):
        return tf.reduce_sum(tf.abs(y_true - y_pred), axis=1) < 0.05

    def success_norm(y_true, y_pred):
        return tf.norm(y_true - y_pred, axis=1) < 0.05

    @tf.function
    def get_energy(basis, vector):
        return tf.norm(tf.matmul(basis, vector), axis=1)

    @tf.function
    def get_energy_changed(basis, vector):
        # print(basis.shape, vector.shape)
        # THIS HAS BIT MINOR CHANGE
        # m => signal_size, n => basis size, b => batch size
        return tf.norm(tf.einsum("mn,bm->bn", basis, vector), axis=1)[:, None]

    @tf.function
    def get_initial_lifting(basis, energy):
        # print(energy.shape)
        """
				example to see the dark magic:
				energy = np.arange(5).reshape(5, 1)
				basis = np.arange(16).reshape(8, 2)
				# shape should be 5 x 8
				print(basis)
				print(energy)
				np.einsum('ba,sd->bs', energy, basis)
				b => batch_size, 
				a => 1, 
				s => signal_size, 
				d => number of basis
				
				so well einsum will multiply energy with each basis and then add them up(yes i checked)
				"""
        return tf.einsum("ba,sd->bs", energy, basis)

    class Subspace(Layer):
        def __init__(self, signal_size=8, subspace_size=1):
            super(Subspace, self).__init__()
            self.signal_size = signal_size
            self.subspace_size = subspace_size
            w_init = tf.random_normal_initializer()
            self.w = tf.Variable(
                initial_value=w_init(shape=(subspace_size, signal_size)),
                trainable=True,
            )

        def call(self, inputs):
            return self.w

        def get_config(self):
            return {
                "w": self.w,
                "subspace_size": self.subspace_size,
                "signal_size": self.signal_size,
            }

        # @classmethod
        # def from_config(cls, config):
        #     return cls(**config)

    class QR(Layer):
        def __init__(self, signal_size=8):
            super(QR, self).__init__()
            self.signal_size = signal_size

        def call(self, inputs, weights):
            # print(weights.shape)
            # print('inputs=>> ', inputs)
            # print('weights=>> ', weights)
            q, _ = tf.linalg.qr(tf.transpose(weights))
            self.subspaces = q  # saving subspaces to watch later
            energy = get_energy_changed(q, inputs)
            # print(energy)
            # return energy
            # energy = energy/tf.norm(energy, axis=1)
            # print(energy)
            no_of_basis = q.shape[1]
            energy = energy / np.sqrt(no_of_basis)
            lifted = get_initial_lifting(q, energy)
            # print('Lifted=>> ', lifted.shape)
            return lifted

        def get_config(self):
            return {}
            # return { "subspaces": self.subspaces }

        @classmethod
        def from_config(cls, config):
            return cls(**config)

    #################################################################
    ##############Start RUNNING PART#################################
    #################################################################

    subspaces = 50
    signal_size = 8
    X = np.random.rand(10_000, 8)
    X = X / LA.norm(X, axis=1)[:, np.newaxis]
    X_train = X[:7000]
    X_test = X[7000:]
    EPOCHS = 3000

    for subspaces in range(2, 51, 2):
        input_layer = Input(shape=(signal_size))

        subspace_layers = []
        subspace_basis = [random.randint(1, signal_size + 1) for _ in range(subspaces)]

        print(f"SUBSPACE_BASIS FOR {subspaces} subspaces=> ", subspace_basis)
        for i in range(subspaces):
            basis = subspace_basis[i]
            subspace = Subspace(signal_size, basis)(input_layer)
            qr = QR()(input_layer, subspace)
            subspace_layers.append(qr)

        merged = Concatenate()(subspace_layers)
        merged_flatten = Reshape((subspaces * signal_size,))(merged)
        y = Dense(signal_size, activation="sigmoid")(merged_flatten)
        model = Model(inputs=input_layer, outputs=y)
        model.compile(
            optimizer='adam', 
            loss='mean_squared_error',
            metrics=[
                avg_dist,
                avg_norm,
                success,
                success_norm
            ])
        # old_weights_path = f"./v0_5000/weights_{subspaces}_v1/cp.ckpt" 
        checkpoint_path = f"./v1_3000/weights_{subspaces}_v1/cp.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path, 
            save_weights_only=True, 
            verbose=0)
        # model.load_weights(old_weights_path)
        # model.compile(
        #     optimizer="adam",
        #     loss="mean_squared_error",
        #     metrics=[avg_dist, avg_norm, success, success_norm],
        #     callbacks=[cp_callback], 
        #     verbose = 2
        # ) 
        history = model.fit(
            X_train, 
            X_train, 
            epochs=EPOCHS, 
            batch_size=128, 
            callbacks=[cp_callback],
            verbose=2
            )
        _, avg_norm_of_signal, avg_distance, success_, success_norm_ = model.evaluate(X_test, X_test, batch_size=128)
        print(f'{subspaces} =>> {avg_norm_of_signal},    {success_norm_}')
#   model.summary()

