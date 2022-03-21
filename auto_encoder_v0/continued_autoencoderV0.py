import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"    
import numpy as np
from numpy import random, linalg as LA
from keras.layers import Dense, Layer, Input, Concatenate
from keras import Model

import tensorflow as tf
# my_devices = tf.config.experimental.list_physical_devices(device_type='CPU')
# tf.config.experimental.set_visible_devices(devices= my_devices, device_type='CPU')
# tf.debugging.set_log_device_placement(True)
VECTOR_SIZE = 8

device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    device_name = "/cpu:0"
else:
    device_name = "/gpu:0"
device_name = "/cpu:0"
with tf.device(device_name):
# if True:
    def avg_dist(y_true, y_pred):
        return tf.reduce_sum(tf.abs(y_true-y_pred), axis=1)

    def avg_norm(y_true, y_pred):
        return tf.norm(y_true-y_pred, axis=1)

    def success(y_true, y_pred):
        return tf.reduce_sum(tf.abs(y_true-y_pred), axis=1) < 0.05

    def success_norm(y_true, y_pred):
        return tf.norm(y_true-y_pred, axis=1) < 0.05

    def projected_vector(basis, vector):
        return np.matmul(basis.T, vector)

    def projection(basis, vector):
        return LA.norm(projected_vector(basis, vector))

    @tf.function
    def get_energy(basis, vector):
        return tf.norm(tf.matmul(basis, vector), axis=1)

    """
    Parrallel subspaces of different size, SO NO MORE SAME SIZED 3d kernal stuff!!
    WITH QR!!!!
    """
    @tf.function
    def get_energy_changed(basis, vector):
        # print(basis.shape, vector.shape)
        # THIS HAS BIT MINOR CHANGE
        # m => signal_size, n => basis size, b => batch size
        return tf.norm(tf.einsum('mn,bm->bn', basis, vector), axis=1)[:, None] 

    class Subspace(Layer):
        def __init__(self, signal_size=8, subspace_size=1):
            super(Subspace, self).__init__()
            self.signal_size=signal_size
            self.subspace_size = subspace_size
            # seed=10
            w_init = tf.random_normal_initializer()
            self.w = tf.Variable(
                initial_value=w_init(shape=(subspace_size, signal_size), dtype="float32"),
                trainable=True,
            )
        def call(self, inputs):
            return self.w
        
        def get_config(self):
            return {"hidden_units": self.hidden_units}
            # return get_energy(self.w, inputs)

    class QR(Layer):
        def __init__(self, signal_size=8):
            super(QR, self).__init__()
            self.signal_size = signal_size
            # self.subspaces = None

        def call(self, inputs, weights):
            # print(weights.shape)
            q, _ = tf.linalg.qr(tf.transpose(weights))
            self.subspaces = q # saving subspaces to watch later
            return get_energy_changed(q, inputs)

        def get_config(self):
            return {"hidden_units": self.hidden_units}
    np.random.seed(42)
    X = np.random.rand(10_000, 8)
    X = X/LA.norm(X, axis=1)[:, np.newaxis]
    X_train = X[:7000]
    X_test = X[7000:]
    EPOCHS = 3000
    batch_size = 128
    signal_size = 8
    for subspaces in range(2, 52, 2):
        x = tf.random.uniform((signal_size,))
        input_layer = Input(shape=(signal_size))
        subspace_layers = []
        for i in range(subspaces):
            basis = random.randint(1, signal_size+1)
            subspace = Subspace(signal_size, basis)(input_layer)
            qr = QR()(input_layer, subspace)
            subspace_layers.append(qr)
        # if subspaces!=50:
        #     continue
        merged = Concatenate()(subspace_layers)
        y = Dense(8, input_dim=signal_size, activation='sigmoid')(merged)
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
        old_weights_path = f"./v0_5000/weights_{subspaces}_v1/cp.ckpt" 
        checkpoint_path = f"./v0_8000/weights_{subspaces}_v1/cp.ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path, save_weights_only=True, verbose=0)
        model.load_weights(old_weights_path)
        history = model.fit(X_train, X_train, epochs=EPOCHS, batch_size=batch_size, callbacks=[cp_callback], verbose = 2)
        _, avg_norm_of_signal, avg_distance, success_, success_norm_ = model.evaluate(X_test, X_test, batch_size=batch_size)
        print(f'{subspaces} =>> {avg_norm_of_signal},    {success_norm_}')