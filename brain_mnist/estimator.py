import tensorflow as tf

from brain_mnist.model import Resnet10


class BrainMnistEstimator(object):
    def __init__(self, params):
        self._instantiate_model(params)

    def _instantiate_model(self, params, training=False):
        if params['model'] == 'resnet10':
            self.model = Resnet10(params=params, is_training=training)

    def _output_network(self, features, params, training=False):
        self._instantiate_model(params=params, training=training)
        output = self.model(inputs=features)
        return output

    def loss_fn(self, labels, predictions, params):
        digit_pred = predictions['digit']
        digit_label = labels['digit_label']

        losses = tf.nn.softmax_cross_entropy_with_logits_v2(labels=digit_label, logits=digit_pred)
        loss = tf.reduce_sum(losses) / params['batch_size']
        tf.summary.scalar('loss', tensor=loss)
        return loss

    def model_fn(self, features, labels, mode, params):
        training = (mode == tf.estimator.ModeKeys.TRAIN)

        preds = self._output_network(features, params, training=training)

        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.AdamOptimizer(learning_rate=params['lr'])

            predictions = {'digit': preds}
            loss = self.loss_fn(labels, predictions, params)
            train_op = tf.contrib.training.create_train_op(loss, optimizer, global_step=tf.train.get_global_step())

            pred_val = tf.one_hot(tf.argmax(tf.nn.softmax(predictions['digit']), -1), params['n_classes'])
            acc = tf.metrics.accuracy(labels=labels['digit_label'], predictions=pred_val)
            tf.summary.scalar('acc', tensor=acc[1], family='accuracy')
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        if mode == tf.estimator.ModeKeys.EVAL:
            predictions = {'digit': preds}
            pred_val = tf.one_hot(tf.argmax(tf.nn.softmax(predictions['digit']), -1), params['n_classes'])
            metrics = {
                'accuracy/accuracy/acc': tf.metrics.accuracy(labels=labels['digit_label'], predictions=pred_val)
            }

            loss = self.loss_fn(labels, predictions, params)
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=metrics)

        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                'signal_input': features['signal_input'],
                'digit': tf.nn.softmax(preds)
            }
            export_outputs = {'predictions': tf.estimator.export.PredictOutput(predictions)}
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, export_outputs=export_outputs)
