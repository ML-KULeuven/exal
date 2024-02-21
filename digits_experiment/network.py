import bisect
import tensorflow as tf

from keras.layers import *

class DigitNet(tf.keras.Model):

    def __init__(self, digits, batch_size=1, learning_rate=1e-4):
        super().__init__()
        self.digits = digits
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)
        self.logger = Logger()

        self.model = tf.keras.Sequential()
        self.model.add(Conv2D(6, 5, activation='relu'))
        self.model.add(MaxPooling2D(2, 2))
        self.model.add(Conv2D(16, 5, activation='relu'))
        self.model.add(MaxPooling2D(2, 2))
        self.model.add(Flatten())
        self.model.add(Dense(120, activation='relu'))
        self.model.add(Dense(84, activation='relu'))
        self.model.add(Dense(10, activation='linear'))

    def call(self, x):
        x = tf.concat([x[:, i, :, :, :] for i in range(self.digits)], axis=0)
        x = self.model(x)
        return tf.stack([x[i * self.batch_size:(i + 1) * self.batch_size, :] for i in range(self.digits)], axis=1)
    
    @tf.function
    def elbo_grads(self, n1, n2, n1_samples, n2_samples, loss):
        with tf.GradientTape() as tape:
            n1_probs = self(n1)
            n2_probs = self(n2)
            n1_probs = tf.expand_dims(n1_probs, axis=1)
            n2_probs = tf.expand_dims(n2_probs, axis=1)
            n1_probs = tf.repeat(n1_probs, n1_samples.shape[1], axis=1)
            n2_probs = tf.repeat(n2_probs, n2_samples.shape[1], axis=1)

            n1_sample_probs = tf.gather(n1_probs, n1_samples, batch_dims=3, axis=-1)[..., 0]
            n2_sample_probs = tf.gather(n2_probs, n2_samples, batch_dims=3, axis=-1)[..., 0]

            if loss == 'no_agree':
                sample_probs = tf.reduce_prod(tf.concat([n1_sample_probs, n2_sample_probs], axis=-1), axis=-1)
                sample_probs = -tf.math.log(sample_probs + 1e-8)
                mean_logprob = tf.reduce_mean(sample_probs, axis=-1)
                loss = tf.reduce_mean(mean_logprob)
            else:
                sumstuff = tf.reduce_sum(tf.math.log(tf.concat([n1_sample_probs, n2_sample_probs], axis=-1) + 1e-8), axis=-1)
                batch_loss = -tf.reduce_logsumexp(sumstuff, axis=-1)

                loss = tf.reduce_mean(batch_loss)
            
            return loss, tape.gradient(loss, self.trainable_variables)
        
    @tf.function
    def corrected_elbo_grads(self, n1, n2, n1_samples, n2_samples, loss, weights):
        with tf.GradientTape() as tape:
            n1_probs = self(n1)
            n1_probs = tf.nn.softmax(n1_probs, axis=-1)
            n2_probs = self(n2)
            n2_probs = tf.nn.softmax(n2_probs, axis=-1)
            n1_probs = tf.expand_dims(n1_probs, axis=1)
            n2_probs = tf.expand_dims(n2_probs, axis=1)
            n1_probs = tf.repeat(n1_probs, n1_samples.shape[1], axis=1)
            n2_probs = tf.repeat(n2_probs, n2_samples.shape[1], axis=1)

            n1_sample_probs = tf.gather(n1_probs, n1_samples, batch_dims=3, axis=-1)[..., 0]
            n2_sample_probs = tf.gather(n2_probs, n2_samples, batch_dims=3, axis=-1)[..., 0]

            if loss == 'no_agree':
                sample_probs = tf.concat([n1_sample_probs, n2_sample_probs], axis=-1)
                log_sample_probs = -tf.math.log(sample_probs + 1e-8)
                log_probs = tf.reduce_sum(log_sample_probs, axis=-1) / weights
                mean_logprob = tf.reduce_mean(log_probs, axis=-1)
                loss = tf.reduce_mean(mean_logprob)
            else:
                sumstuff = tf.reduce_sum(tf.math.log(tf.concat([n1_sample_probs, n2_sample_probs], axis=-1) + 1e-8), axis=-1)
                weighted_sumstuff = sumstuff - tf.math.log(weights + 1e-8)
                batch_loss = -tf.reduce_logsumexp(weighted_sumstuff, axis=-1)
                loss = tf.reduce_mean(batch_loss)
            
            return loss, tape.gradient(loss, self.trainable_variables)

class Logger(object):

    def __init__(self):
        super(Logger, self).__init__()
        self.log_dict = dict()
        self.indices = list()

    def log(self, name, index, value):
        if name not in self.log_dict:
            self.log_dict[name] = dict()
        i = bisect.bisect_left(self.indices, index)
        if i >= len(self.indices) or self.indices[i] != index:
            self.indices.insert(i, index)
        self.log_dict[name][index] = value
