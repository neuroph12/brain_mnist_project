import os
import multiprocessing
from absl import flags, app
from copy import deepcopy

from tqdm import tqdm
import tensorflow as tf
import numpy as np

from brain_mnist.data_processing.mindwave_parser import parse_data_file


flags.DEFINE_string('data_dir',
                    '/home/filippo/datasets/mindwave/',
                    'data directory path')
flags.DEFINE_string('data_split',
                    '0.7/0.15',
                    'train/val split')
flags.DEFINE_integer('num_shards',
                     256,
                     'number of tfrecord files')
flags.DEFINE_boolean('debug',
                     False,
                     'debug for a few samples')
flags.DEFINE_string('data_type',
                    'trainvaltest',
                    'data types to write into tfrecords')

FLAGS = flags.FLAGS
tf.logging.set_verbosity(tf.logging.INFO)


def int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def float_list_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


def create_tf_example(entry_id, entry_event, device, channel, code, size, signal):
    feature_dict = {
        'id': int64_feature(int(entry_id)),
        'event': int64_feature(int(entry_event)),
        'device': bytes_feature(device.encode()),
        'channel': bytes_feature(channel.encode()),
        'code': int64_feature(int(code)),
        'size': int64_feature(int(size)),
        'signal': float_list_feature(signal.tolist())
    }

    example = tf.train.Example(features=tf.train.Features(feature=feature_dict))
    return example


def pool_create_tf_example(args):
    return create_tf_example(*args)


def write_tfrecords(path, dataiter, num_shards=256, nmax=-1):
    writers = [
        tf.python_io.TFRecordWriter('{}{:05d}_{:05d}.tfrecord'.format(path, i, num_shards)) for i in range(num_shards)
    ]
    print('\nWriting to output path: {}'.format(path))
    pool = multiprocessing.Pool()
    for i, tf_example in tqdm(enumerate(pool.imap(pool_create_tf_example, [(deepcopy(data['id']),
                                                                            deepcopy(data['event']),
                                                                            deepcopy(data['device']),
                                                                            deepcopy(data['channel']),
                                                                            deepcopy(data['code']),
                                                                            deepcopy(data['size']),
                                                                            deepcopy(data['signal'])
                                                                            ) for data in dataiter]))):
        if tf_example is not None:
            writers[i % num_shards].write(tf_example.SerializeToString())
        if 0 < nmax < i:
            break
    pool.close()
    for writer in writers:
        writer.close()


def make_split(meta_data, data, split='0.7/0.15'):
    meta_data = np.array(meta_data)

    train_prop, val_prop = [float(x) for x in split.split('/')]
    test_prop = 1.0 - train_prop - val_prop
    assert test_prop < 1.0, 'Wrong split flag'

    num_samples = len(data)
    num_train, num_val = int(train_prop * num_samples), int(val_prop * num_samples)

    inds = [i for i in range(num_samples)]
    np.random.shuffle(inds)
    train_inds, val_inds, test_inds = inds[:num_train], inds[num_train:num_train + num_val], inds[num_train + num_val:]

    train = [meta_data[train_inds], data[train_inds]]
    val = [meta_data[val_inds], data[val_inds]]
    test = [meta_data[test_inds], data[test_inds]]
    return train, val, test


def data_iterator(dataset):
    meta_data, data = dataset
    for i in range(len(data)):
        yield_dict = {
            'id': meta_data[i][0],
            'event': meta_data[i][1],
            'device': meta_data[i][2],
            'channel': meta_data[i][3],
            'code': meta_data[i][4],
            'size': meta_data[i][5],
            'signal': data[i]
        }
        yield yield_dict


def create_tfrecords(data_dir,
                     split='0.7/0.15',
                     num_shards=256,
                     debug=False,
                     data_type='trainval'):
    np.random.seed(0)

    output_path = os.path.join(data_dir, 'tfrecords/')
    if not tf.gfile.IsDirectory(output_path):
        tf.gfile.MakeDirs(output_path)

    fp = os.path.join(data_dir, 'MW.txt')
    meta_data, data = parse_data_file(fp)

    train, val, test = make_split(meta_data, data, split=split)

    print('\nTotal signals: {}'.format(len(train[1]) + len(val[1]) + len(test[1])))
    print('Train/val/test {} split: {}/{}/{}'.format(split, len(train[1]), len(val[1]), len(test[1])))
    train_it = data_iterator(train)
    val_it = data_iterator(val)
    test_it = data_iterator(test)

    nmax = 10 if debug else -1
    if 'train' in data_type:
        print('\nWriting train tfrecords ...')
        train_path = os.path.join(output_path, 'train/')
        if not tf.gfile.IsDirectory(train_path):
            tf.gfile.MakeDirs(train_path)
        write_tfrecords(train_path, train_it, num_shards, nmax=nmax)

    if 'val' in data_type:
        print('\nWriting val tfrecords ...')
        val_path = os.path.join(output_path, 'val/')
        if not tf.gfile.IsDirectory(val_path):
            tf.gfile.MakeDirs(val_path)
        write_tfrecords(val_path, val_it, num_shards, nmax=nmax)

    if 'test' in data_type:
        print('\nWriting test tfrecords ...')
        test_path = os.path.join(output_path, 'test/')
        if not tf.gfile.IsDirectory(test_path):
            tf.gfile.MakeDirs(test_path)
        write_tfrecords(test_path, test_it, num_shards, nmax=nmax)


def main(_):
    create_tfrecords(FLAGS.data_dir,
                     split=FLAGS.data_split,
                     num_shards=FLAGS.num_shards,
                     debug=FLAGS.debug,
                     data_type=FLAGS.data_type)


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    app.run(main)
