import time
import wandb
import numpy as np
import tensorflow as tf
import tensorflow_probability.python.distributions as tfd

from digits_experiment.network import DigitNet, Logger
from digits_experiment.evaluate import mnist_sum_test


class SumSampler(tf.keras.Model):

    def __init__(self, digits, samples, loss='', alpha=0., batch_size=10, learning_rate=1e-3, annealed=False):
        super().__init__()
        self.digits = digits
        self.samples = samples
        self.loss = loss
        self.alpha = alpha
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.annealed = annealed
        self.model = DigitNet(self.digits, batch_size=self.batch_size)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.learning_rate)

        if self.digits >= 10:
            self.type = tf.int64
        else:
            self.type = tf.int32

        self.init_multipliers()
        self.init_counter()
        self.acc_logits = 0
        self.acc_counts = 0

        self.logger = Logger()

    def init_multipliers(self):
        multiplier = []
        for i in range(self.digits):
            multiplier.append(tf.ones((self.batch_size, self.samples), dtype=self.type) * 10 ** i)
        self.multiplier = tf.stack(multiplier[::-1], axis=-1)

        multiplier2 = []
        for i in range(self.digits - 1):
            multiplier2.append(tf.ones((self.batch_size, self.samples), dtype=self.type) * 10)
        multiplier2.append(tf.zeros((self.batch_size, self.samples), dtype=self.type))
        self.multiplier2 = tf.stack(multiplier2, axis=-1)

    def init_counter(self):
        # self.sample_counters = tf.Variable(tf.ones([1, self.digits, 10], dtype=np.float32), trainable=False)
        self.sample_counters = {i: tf.Variable(tf.ones([10], dtype=np.float32), trainable=False) for i in range(self.digits)}

    # @tf.function
    def call(self, n1, s):
        n1_probs = self.model(n1) # It outputs logits for the first multidigit number [BATCH, self.digits, 10]
        constraint = tf.zeros([self.batch_size, self.samples], dtype=tf.int32)

        n1_samples = []
        for i in range(self.digits):
            if i == 0:
                maxim = tf.cast(s // (10 ** (self.digits - i - 1)), dtype=tf.int32)
                minim = tf.maximum(0, maxim - 9)
            else:
                maxim = tf.cast(s // (10 ** (self.digits - i - 1)) % 10, dtype=tf.int32)
                minim = tf.zeros_like(maxim)

            # if i == 0:
            #     maxim = s // (10 ** (self.digits - i - 1))
            #     minim = tf.maximum(0, maxim - 9)
            # else:
            #     maxim = s // (10 ** (self.digits - i - 1)) % 10
            #     minim = tf.zeros_like(maxim, dtype=tf.int64)

            probs = tf.repeat(tf.expand_dims(n1_probs[:, i, ...], 1), self.samples, axis=1) # [BATCH, self.samples, 10] logits for single digit considering multiple samples

            self.acc_logits += tf.reduce_mean(probs)
            # self.acc_counts += tf.reduce_mean(tf.math.log(self.sample_counters[:, i:i+1, :]))

            sample_counter = tf.expand_dims(tf.expand_dims(self.sample_counters[i], axis=0), 0)
            self.acc_counts += tf.reduce_mean(tf.math.log(sample_counter))
            
            # probs = probs - self.alpha * tf.math.log(self.sample_counters[:, i:i+1, :])
            probs = probs - self.alpha * tf.math.log(sample_counter)
            probs_cut = tf.concat([tf.concat([-np.inf * tf.ones([1, self.samples, tf.maximum(0, minim[j])]), probs[j:j+1, :, minim[j]:maxim[j] + 1], -np.inf * tf.ones([1, self.samples, tf.maximum(0, 10 - maxim[j] - 1)])], axis=-1) for j in range(self.batch_size)], axis=0)
            probs = tf.where(tf.expand_dims(constraint, axis=-1) == 1, probs, probs_cut)

            d = tfd.Categorical(logits=probs)
            samples = d.sample(1) # [1, BATCH, self.samples]
            samples = samples[0, ...] # [BATCH, self.samples]

            constraint = tf.maximum(tf.where(samples == tf.expand_dims(maxim, 1), 0, 1), constraint)
            n1_samples.append(tf.cast(samples, dtype=self.type))
            self.sample_counters[i].assign_add(tf.reduce_sum(tf.one_hot(samples, 10), axis=[0, 1]))
            # self.sample_counters[:, i, :] = self.sample_counters[:, i, :] + tf.reduce_sum(tf.one_hot(samples, 10), axis=[0, 1])

        self.acc_logits = self.acc_logits / self.digits
        self.acc_counts = self.acc_counts / self.digits

        n1_samples = tf.stack(n1_samples, axis=-1) # [BATCH, self.samples, self.digits]
        n1_summed_samples = tf.reduce_sum(n1_samples * self.multiplier, -1)
        n2_samples = tf.expand_dims(s, axis=-1) - n1_summed_samples # Computing the second multi-digit number

        weights = self.sample_weights(n1_summed_samples)
        n2_samples = self.decimalise(n2_samples)

        n1_samples = tf.expand_dims(n1_samples, axis=-1)
        n2_samples = tf.expand_dims(n2_samples, axis=-1)

        return n1_samples, n2_samples, weights
    
    @tf.function
    def decimalise(self, n2_samples):
        n2_samples = tf.repeat(tf.expand_dims(n2_samples, -1), self.digits, axis=-1)
        n2_samples = n2_samples // self.multiplier
        n2_samples -= tf.sort(n2_samples * self.multiplier2, axis=-1)
        return n2_samples
    
    @tf.function
    def sample_weights(self, n1_summed_samples):
        weights = []
        for j in range(self.batch_size):
            _, idx, counts = tf.unique_with_counts(n1_summed_samples[j, ...])
            weights.append(tf.cast(tf.gather(counts, idx, batch_dims=0, axis=-1), dtype=tf.float32))
        return tf.stack(weights, axis=0)

    def train(self, train_dataset, val_dataset, test_dataset, epochs, ):
        count = 0
        for e in range(epochs):
            if self.annealed:
                self.alpha = self.alpha * 0.95
            print(self.alpha)
            acc_loss = 0
            time_start = time.time()
            for n1, n2, s in train_dataset:
                n1_samples, n2_samples, weights = self(n1, s)

                loss, grads = self.model.corrected_elbo_grads(n1, n2, n1_samples, n2_samples, loss=self.loss, weights=weights)
                acc_loss += loss
                self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
                count += 1

                if count % 100 == 0:
                    end = time.time()
                    val_acc = mnist_sum_test(val_dataset, self.model)
                    test_acc = mnist_sum_test(test_dataset, self.model)
                    print(
                        f"Epoch: {e + 1}",
                        f"Iteration: {count}",
                        f"Loss: {round(acc_loss.numpy() / 100, 5)}",
                        f"Logits: {round(self.acc_logits.numpy() / 100, 5)}",
                        f"Counts: {round(self.acc_counts.numpy() / 100, 5)}",
                        f"Val accuracy: {round(val_acc.numpy() * 100, 3)}%",
                        f"Test accuracy: {round(test_acc.numpy() * 100, 3)}%",
                        f"Time: {round(end - time_start, 3)}s",             
                    )
                    self.logger.log("loss", count, acc_loss / 100)
                    self.logger.log("val_acc", count, test_acc)
                    self.logger.log("test_acc", count, test_acc)
                    self.logger.log("logits", count, self.acc_logits)
                    self.logger.log("counts", count, self.acc_counts)
                    self.logger.log("time", count, end - time_start)
                    wandb.log({
                        "Loss": round(acc_loss.numpy() / 100, 5),
                        "Logits": round(self.acc_logits.numpy() / 100, 5),
                        "Counts": round(self.acc_counts.numpy() / 100, 5),
                        "Val Accuracy": round(val_acc.numpy() * 100, 3), 
                        "Test Accuracy": round(test_acc.numpy() * 100, 3), 
                        "Time": round(end - time_start, 3)})
                    acc_loss = 0
                    self.acc_logits = 0
                    self.acc_counts = 0
                    time_start = time.time()
