
import numpy as np
import tensorflow as tf

def number_multiplier(batch_size, digits):
    multiplier = np.ones((batch_size, digits))
    for i in range(digits):
        multiplier[:, i] = 10 ** (digits - i - 1)
    return multiplier

def mnist_sum_test(test_set, model):
    correct = 0
    total = 0
    multiplier = number_multiplier(10, model.digits)
    for x1, x2, y in test_set:
        n1_probs = tf.squeeze(model(x1))
        n2_probs = tf.squeeze(model(x2))

        predicted_d1 = tf.math.argmax(n1_probs, axis=-1) * multiplier
        predicted_d2 = tf.math.argmax(n2_probs, axis=-1) * multiplier
        predicted_digits = tf.concat([predicted_d1, predicted_d2], axis=-1)

        predicted_sum = tf.cast(tf.math.reduce_sum(predicted_digits, axis=-1), y.dtype)
        correct += tf.math.count_nonzero(tf.math.equal(predicted_sum, y))
        total += y.shape[0]
    return correct / total
