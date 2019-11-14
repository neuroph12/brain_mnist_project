import os
import argparse
import glob

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from brain_mnist.data_processing.mindwave_parser import MW_FREQ
# MAX_SIZE = 1024
MAX_SIZE = 512


def get_dataset(tfrecords,
                batch_size,
                epochs,
                input_size=MAX_SIZE,
                n_classes=10,
                shuffle=False,
                fake_input=False):
    def parse_func(example_proto):
        feature_dict = {
            'id': tf.FixedLenFeature([], tf.int64),
            'event': tf.FixedLenFeature([], tf.int64),
            'device': tf.FixedLenFeature([], tf.string),
            'channel': tf.FixedLenFeature([], tf.string),
            'code': tf.FixedLenFeature([], tf.int64),
            'size': tf.FixedLenFeature([], tf.int64),
            'signal': tf.FixedLenFeature([input_size], tf.float32),
        }

        parsed_feature = tf.parse_single_example(example_proto, feature_dict)

        features, labels = {}, {}
        for key, val in parsed_feature.items():
            if key == 'signal':
                val = tf.cast(val, dtype=tf.float32)
                # val = tf.subtract(val, tf.reduce_min(val, axis=-1, keep_dims=True))
                # val = tf.divide(val, tf.reduce_max(val, axis=-1, keep_dims=True))
                features[key] = val
            elif key == 'code':
                val = tf.one_hot(val, depth=n_classes)
                val = tf.cast(val, dtype=tf.float32)
                labels[key] = val
        return features, labels

    files = tf.data.Dataset.list_files(tfrecords)
    dataset = files.apply(tf.contrib.data.parallel_interleave(tf.data.TFRecordDataset, cycle_length=10))
    if shuffle:
        dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=256, count=epochs))
    else:
        dataset = dataset.repeat(epochs)
    dataset = dataset.map(parse_func, num_parallel_calls=8)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=256)
    if fake_input:
        dataset = dataset.take(1).cache().repeat()
    return dataset


def data_input_fn(tfrecords,
                  batch_size,
                  epochs,
                  input_size=MAX_SIZE,
                  n_classes=10,
                  shuffle=False,
                  fake_input=False):
    def _input_fn():
        dataset = get_dataset(tfrecords,
                              batch_size,
                              epochs,
                              input_size,
                              n_classes,
                              shuffle,
                              fake_input)

        it = dataset.make_one_shot_iterator()
        next_batch = it.get_next()

        signal_input = next_batch[0]['signal']
        digit_label = next_batch[1]['code']

        # Scale input signal to [0, 1]
        signal_input = tf.subtract(signal_input, tf.reduce_min(signal_input, axis=-1, keep_dims=True))
        signal_input = tf.divide(signal_input, tf.reduce_max(signal_input, axis=-1, keep_dims=True))
        signal_input = tf.expand_dims(signal_input, axis=-1)

        features, labels = {}, {}
        features['signal_input'] = signal_input
        labels['digit_label'] = digit_label
        return features, labels
    return _input_fn


def main():
    parser = argparse.ArgumentParser(description='debug input pipeline')
    parser.add_argument('--data-dir', '-d', type=str, default='/home/filippo/datasets/mindwave/tfrecords/')
    parser.add_argument('--n-classes', '-n', type=int, default=10)
    parser.add_argument('--data-type', '-t', type=str, default='train')
    args = parser.parse_args()

    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    tfrecords = glob.glob('{}{}/*.tfrecord'.format(args.data_dir, args.data_type))
    dataset = get_dataset(tfrecords,
                          batch_size=32,
                          epochs=1,
                          input_size=MAX_SIZE,
                          n_classes=args.n_classes,
                          shuffle=True,
                          fake_input=False)
    print('\nDataset out types {}'.format(dataset.output_types))

    batch = dataset.make_one_shot_iterator().get_next()
    sess = tf.Session()
    try:
        batch_nb = 0
        while True:
            data = sess.run(batch)
            batch_nb += 1

            signal_input = data[0]['signal']
            digit_label = data[1]['code']

            print('\nBatch nb {}'.format(batch_nb))
            for i in range(len(signal_input)):
                print('Digit: {}'.format(digit_label[i]))
                sns.set()
                sns.lineplot(x=[i / MW_FREQ for i in range(MAX_SIZE)], y=signal_input[i])
                plt.title('EEG signal corresponding to digit {}'.format(np.argmax(digit_label[i])))
                plt.xlabel('Time (s)')
                plt.ylabel('EEG signal amplitude')
                plt.show()

    except tf.errors.OutOfRangeError:
        pass


if __name__ == '__main__':
    main()
