import os
import glob
import argparse
import logging
import sys
from datetime import datetime

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns

from brain_mnist.input_pipeline import data_input_fn
from brain_mnist.estimator import BrainMnistEstimator


def main():
    parser = argparse.ArgumentParser(description='train cnn for EEG classification')
    parser.add_argument('--data-dir', '-d', type=str, default='/home/filippo/datasets/mindwave/tfrecords/',
                        help='tf records data directory')
    parser.add_argument('--model-dir', type=str, default='', help='pretrained model directory')
    parser.add_argument('--ckpt', type=str, default='', help='pretrained checkpoint directory')
    parser.add_argument('--mode', '-m', type=str, default='train', help='train, eval or predict')
    parser.add_argument('--model', type=str, default='resnet10', help='model name')
    parser.add_argument('--batch-size', '-bs', type=int, default=32, help='batch size')
    parser.add_argument('--epochs', '-e', type=int, default=10, help='train epochs')
    parser.add_argument('--n-filters', type=str, default='32-64-64')
    parser.add_argument('--n-kernels', type=str, default='8-5-3')
    parser.add_argument('--n-classes', '-n', type=int, default=10, help='number of classes')
    parser.add_argument('--learning-rate', '-lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--fake-input', action='store_true', default=False, help='debug with 1 batch training')
    parser.add_argument('--dropout', action='store_true', default=False, help='dropout')
    args = parser.parse_args()
    assert args.model in ['resnet10', 'rnn'], 'Wrong model name'
    assert len(args.n_filters.split('-')) == 3, '3 values required'
    assert len(args.n_kernels.split('-')) == 3, 'Wrong n_filter arg'

    tfrecords_train = glob.glob('{}train/*.tfrecord'.format(args.data_dir))
    tfrecords_val = glob.glob('{}val/*.tfrecord'.format(args.data_dir))
    tfrecords_test = glob.glob('{}test/*.tfrecord'.format(args.data_dir))

    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    logger = logging.getLogger(__name__)
    np.random.seed(0)
    tf.set_random_seed(0)
    tf.logging.set_verbosity(tf.logging.INFO)

    if not args.model_dir:
        save_dir = '{}models/{}/{}/'.format(args.data_dir, args.model, datetime.now().isoformat())
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    else:
        save_dir = args.model_dir

    params = {
        'model': args.model,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'n_cnn_filters': [int(x) for x in args.n_filters.split('-')],
        'n_cnn_kernels': [int(x) for x in args.n_kernels.split('-')],
        'n_classes': args.n_classes,
        'lr': args.learning_rate,
        'drop': args.dropout,
    }

    train_config = tf.estimator.RunConfig(save_summary_steps=10,
                                          save_checkpoints_steps=500,
                                          keep_checkpoint_max=20,
                                          log_step_count_steps=10)

    ws = None
    if args.ckpt:
        ws = tf.estimator.WarmStartSettings(ckpt_to_initialize_from=args.ckpt, vars_to_warm_start='.*')

    estimator_obj = BrainMnistEstimator(params)
    estimator = tf.estimator.Estimator(model_fn=estimator_obj.model_fn,
                                       model_dir=save_dir,
                                       config=train_config,
                                       params=params,
                                       warm_start_from=ws)

    mode_keys = {
        'train': tf.estimator.ModeKeys.TRAIN,
        'eval': tf.estimator.ModeKeys.EVAL,
        'predict': tf.estimator.ModeKeys.PREDICT
    }
    mode = mode_keys[args.mode]
    if mode == tf.estimator.ModeKeys.TRAIN:
        train_input_fn = data_input_fn(tfrecords_train,
                                       batch_size=params['batch_size'],
                                       epochs=1,
                                       n_classes=params['n_classes'],
                                       shuffle=True,
                                       fake_input=args.fake_input)
        eval_input_fn = data_input_fn(tfrecords_val,
                                      batch_size=params['batch_size'],
                                      epochs=1,
                                      n_classes=params['n_classes'],
                                      shuffle=False,
                                      fake_input=args.fake_input)

        for epoch_num in range(params['epochs']):
            logger.info("Training for epoch {} ...".format(epoch_num))
            estimator.train(input_fn=train_input_fn)
            logger.info("Evaluation for epoch {} ...".format(epoch_num))
            estimator.evaluate(input_fn=eval_input_fn)
    elif mode == tf.estimator.ModeKeys.EVAL:
        test_input_fn = data_input_fn(tfrecords_test,
                                      batch_size=params['batch_size'],
                                      epochs=1,
                                      n_classes=params['n_classes'],
                                      shuffle=False,
                                      fake_input=args.fake_input)

        logger.info("Evaluation of test set ...")
        estimator.evaluate(input_fn=test_input_fn)
    elif mode == tf.estimator.ModeKeys.PREDICT:
        test_input_fn = data_input_fn(tfrecords_test,
                                      batch_size=params['batch_size'],
                                      epochs=1,
                                      n_classes=params['n_classes'],
                                      shuffle=False,
                                      fake_input=args.fake_input)

        predictions = estimator.predict(input_fn=test_input_fn)
        for n, pred in enumerate(predictions):
            signal_input = pred['signal_input']
            digit_pred = pred['digit']
            print(digit_pred)

            sns.set()
            sns.lineplot(x=[i for i in range(len(signal_input[:, 0]))], y=signal_input[:, 0])
            plt.title('EEG signal corresponding to digit {}'.format(np.argmax(digit_pred)))
            plt.xlabel('Time (s)')
            plt.ylabel('EEG signal amplitude')
            plt.show()


if __name__ == '__main__':
    main()
