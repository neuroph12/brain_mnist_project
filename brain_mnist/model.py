from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.layers import Conv1D, BatchNormalization, Activation, add, Dense, Flatten


class Resnet10(Model):
    def __init__(self, params=None, is_training=False):
        super(Resnet10, self).__init__()

        self.is_training = is_training
        self.n_cnn_filters = params['n_cnn_filters']
        self.n_cnn_kernels = params['n_cnn_kernels']
        self.n_classes = params['n_classes']

        # Block 1
        self.conv1_1 = Conv1D(self.n_cnn_filters[0],
                              self.n_cnn_kernels[0],
                              activation=None,
                              padding='same',
                              name='conv1_1')
        self.bn1_1 = BatchNormalization(name='batchnorm1_1')
        self.relu1_1 = Activation(activation='relu', name='relu1_1')

        self.conv1_2 = Conv1D(self.n_cnn_filters[0],
                              self.n_cnn_kernels[1],
                              activation=None,
                              padding='same',
                              name='conv1_2')
        self.bn1_2 = BatchNormalization(name='batchnorm1_2')
        self.relu1_2 = Activation(activation='relu', name='relu1_2')

        self.conv1_3 = Conv1D(self.n_cnn_filters[0],
                              self.n_cnn_kernels[2],
                              activation=None,
                              padding='same',
                              name='conv1_3')
        self.bn1_3 = BatchNormalization(name='batchnorm1_3')

        self.shortcut1 = Conv1D(self.n_cnn_filters[0],
                                1,
                                activation=None,
                                padding='same',
                                name='shortcut1')
        self.bn_shortcut1 = BatchNormalization(name='batchnorm_shortcut1')

        self.out_block1 = Activation(activation='relu', name='out_block1')

        # Block 2
        self.conv2_1 = Conv1D(self.n_cnn_filters[1],
                              self.n_cnn_kernels[0],
                              activation=None,
                              padding='same',
                              name='conv2_1')
        self.bn2_1 = BatchNormalization(name='batchnorm2_1')
        self.relu2_1 = Activation(activation='relu', name='relu2_1')

        self.conv2_2 = Conv1D(self.n_cnn_filters[1],
                              self.n_cnn_kernels[1],
                              activation=None,
                              padding='same',
                              name='conv2_2')
        self.bn2_2 = BatchNormalization(name='batchnorm2_2')
        self.relu2_2 = Activation(activation='relu', name='relu2_2')

        self.conv2_3 = Conv1D(self.n_cnn_filters[1],
                              self.n_cnn_kernels[2],
                              activation=None,
                              padding='same',
                              name='conv2_3')
        self.bn2_3 = BatchNormalization(name='batchnorm2_3')

        self.shortcut2 = Conv1D(self.n_cnn_filters[1],
                                1,
                                activation=None,
                                padding='same',
                                name='shortcut2')
        self.bn_shortcut2 = BatchNormalization(name='batchnorm_shortcut2')

        self.out_block2 = Activation(activation='relu', name='out_block2')

        # Block 3
        self.conv3_1 = Conv1D(self.n_cnn_filters[2],
                              self.n_cnn_kernels[0],
                              activation=None,
                              padding='same',
                              name='conv3_1')
        self.bn3_1 = BatchNormalization(name='batchnorm3_1')
        self.relu3_1 = Activation(activation='relu', name='relu3_1')

        self.conv3_2 = Conv1D(self.n_cnn_filters[2],
                              self.n_cnn_kernels[1],
                              activation=None,
                              padding='same',
                              name='conv3_2')
        self.bn3_2 = BatchNormalization(name='batchnorm3_2')
        self.relu3_2 = Activation(activation='relu', name='relu3_2')

        self.conv3_3 = Conv1D(self.n_cnn_filters[2],
                              self.n_cnn_kernels[2],
                              activation=None,
                              padding='same',
                              name='conv3_3')
        self.bn3_3 = BatchNormalization(name='batchnorm3_3')

        self.bn_shortcut3 = BatchNormalization(name='batchnorm_shortcut3')
        self.out_block3 = Activation(activation='relu', name='out_block3')

        # FC
        self.flatten = Flatten(name='flatten')
        self.fc1 = Dense(2048, activation='relu', name='fc1')
        self.fc2 = Dense(self.n_classes, activation=None, name='fc2')

    def call(self, inputs, training=None, mask=None):
        signal_input = inputs['signal_input']
        with tf.name_scope('block1'):
            x = self.conv1_1(signal_input)
            x = self.bn1_1(x)
            x = self.relu1_1(x)

            x = self.conv1_2(x)
            x = self.bn1_2(x)
            x = self.relu1_2(x)

            x = self.conv1_3(x)
            x = self.bn1_3(x)

        shortcut1 = self.shortcut1(signal_input)
        shortcut1 = self.bn_shortcut1(shortcut1)
        x = add([x, shortcut1])
        out_block1 = self.out_block1(x)

        with tf.name_scope('block2'):
            x = self.conv2_1(out_block1)
            x = self.bn2_1(x)
            x = self.relu2_1(x)

            x = self.conv2_2(x)
            x = self.bn2_2(x)
            x = self.relu2_2(x)

            x = self.conv2_3(x)
            x = self.bn2_3(x)

        shortcut2 = self.shortcut2(out_block1)
        shortcut2 = self.bn_shortcut2(shortcut2)
        x = add([x, shortcut2])
        out_block2 = self.out_block2(x)

        with tf.name_scope('block3'):
            x = self.conv3_1(out_block2)
            x = self.bn3_1(x)
            x = self.relu3_1(x)

            x = self.conv3_2(x)
            x = self.bn3_2(x)
            x = self.relu3_2(x)

            x = self.conv3_3(x)
            x = self.bn3_3(x)

        shortcut3 = self.bn_shortcut3(out_block2)
        x = add([x, shortcut3])
        out_block3 = self.out_block3(x)

        x = self.flatten(out_block3)
        x = self.fc1(x)
        output = self.fc2(x)
        return output
