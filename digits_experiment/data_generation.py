
import pickle
import tensorflow as tf

TRAIN_BUF = 60000
VAL_BUF = 1000
TEST_BUF = 9000

def create_numbers(N, data_x, data_y, batch_size=10):
    idx_perm = tf.random.shuffle(tf.range(data_x.shape[0]))

    sum_data1 = []
    sum_data2 = []
    value_data = []
    safety_length = data_x.shape[0] // (2 * N * batch_size) * (2 * N * batch_size)
    for i in range(safety_length // (2 * N)):
        sum_image1 = []
        value1 = 0
        for j in range(N):
            sum_image1.insert(0, tf.expand_dims(data_x[idx_perm[2 * N * i + j]], axis=-1))
            value1 += data_y[idx_perm[2 * N * i + j]] * 10 ** j

        sum_image2 = []
        value2 = 0
        for j in range(N, 2 * N):
            sum_image2.insert(0, tf.expand_dims(data_x[idx_perm[2 * N * i + j]], axis=-1))
            value2 += data_y[idx_perm[2 * N * i + j]] * 10 ** (j - N)

        sum_data1.append(tf.stack(sum_image1, axis=0))
        sum_data2.append(tf.stack(sum_image2, axis=0))
        value_data.append(value1 + value2)

    return tf.stack(sum_data1, axis=0), tf.stack(sum_data2, axis=0), tf.stack(value_data, axis=0)

def create_loader(N, BATCH_SIZE=10):
    TRAIN_BUF = 60000 // (2 * N * BATCH_SIZE) * (BATCH_SIZE)
    VAL_BUF = 1000 // (2 * N * BATCH_SIZE) * (BATCH_SIZE)
    TEST_BUF = 9000 // (2 * N * BATCH_SIZE) * (BATCH_SIZE)
    
    try:
        x_train, y_train = pickle.load(open(f'digits_experiment/data/data_{N}_train_batch{BATCH_SIZE}.pkl', 'rb'))
        x_test, y_test = pickle.load(open(f'digits_experiment/data/data_{N}_test_batch{BATCH_SIZE}.pkl', 'rb'))

        train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(TRAIN_BUF).batch(BATCH_SIZE)
        val_dataset = tf.data.Dataset.from_tensor_slices((x_test[:VAL_BUF], y_test[:VAL_BUF])).shuffle(VAL_BUF).batch(BATCH_SIZE)
        test_dataset = tf.data.Dataset.from_tensor_slices((x_test[VAL_BUF:], y_test[VAL_BUF:])).shuffle(TEST_BUF).batch(BATCH_SIZE)

        return train_dataset, val_dataset, test_dataset
    except:
        pass

    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.astype('float32') / 255. # [60000, 28, 28]
    x_test = x_test.astype('float32') / 255.

    x_train1, x_train2, y_train = create_numbers(N, x_train, y_train) # [7500, 4, 28, 28, 1]
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train1, x_train2, y_train)).shuffle(TRAIN_BUF).batch(BATCH_SIZE)

    x_test1, x_test2, y_test = create_numbers(N, x_test, y_test)
    val_dataset = tf.data.Dataset.from_tensor_slices((x_test1[:VAL_BUF], x_test2[:VAL_BUF], y_test[:VAL_BUF])).shuffle(VAL_BUF).batch(BATCH_SIZE)
    test_dataset = tf.data.Dataset.from_tensor_slices((x_test1[VAL_BUF:], x_test2[VAL_BUF:], y_test[VAL_BUF:])).shuffle(TEST_BUF).batch(BATCH_SIZE)

    pickle.dump([x_train, y_train], open(f'digits_experiment/data/data_{N}_train_batch{BATCH_SIZE}.pkl', 'wb'))
    pickle.dump([x_test, y_test], open(f'digits_experiment/data/data_{N}_test_batch{BATCH_SIZE}.pkl', 'wb'))

    return train_dataset, val_dataset, test_dataset
