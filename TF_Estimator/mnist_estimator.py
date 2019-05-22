# -*- coding: utf-8 -*-
"""Build a CNN using Estimators.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1mcnCvOrj8lCYgCvwnuXzRQ2Yw9q_V64M
"""

from __future__ import absolute_import, division, print_function

import tensorflow as tf
import numpy as np

tf.logging.set_verbosity(tf.logging.INFO)

"""### network architecture

- CNN Layer # 1, 32, 5x5, relu
- Pooling Layer #1, 2x2 filter, maxpooling, 2 stride
- CNN Layer #2, 64, 5x5, relu
- Pooling Layer #2, 2x2 flter, maxpooling, 2 stride
- Dense Layer #1, 1024 neurons + dropout(0.4)
- Dense Layer #2, logits layer, 10 neurons, one for digit target

### tf.layers
- conv2d(): # of filters, filter kernel size, padding, activation function arguments
- max_pooling2d(): filter size, stride as arguments
- dense(): # of neurons and activation functions as arguments
"""

# tensor in, transformed tensor out
def cnn_model_fn(features, labels, mode):
  """ model funcition for CNN"""
  # input layer
  input_layer = tf.reshape(features['x'], [-1, 28, 28, 1]) # reshape to 4d tensor
  
  # cnn layer 1: almost same as keras
  conv1 = tf.layers.conv2d(inputs=input_layer,
                          filters=32,
                          kernel_size=[5,5],
                          padding='same',
                          activation=tf.nn.relu)
  # pooling layer 1
  pool1 = tf.layers.max_pooling2d(inputs=conv1, pool_size=[2,2], strides=2)
  
  # cnn layer 2
  conv2 = tf.layers.conv2d(inputs=pool1,
                          filters=64,
                          kernel_size=[5,5],
                          padding='same',
                          activation=tf.nn.relu)
  # pooling layerr 2
  pool2 = tf.layers.max_pooling2d(inputs=conv2, pool_size=[2,2], strides=2)
  
  # dense layer 1
  pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64]) # really same as keras Flatten
  dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
  dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode == tf.estimator.ModeKeys.TRAIN)
  
  # logits layer
  logits = tf.layers.dense(inputs=dropout, units=10)
  
  # for prediction
  predictions = {
      # generate predictions  for PREDICT and EVAL mode
      "classes": tf.argmax(input=logits, axis=1),
      # add softmax_tensor to the graph, it is used for PREDICT and by the logging_hook
      "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
  }
  
  if mode == tf.estimator.ModeKeys.PREDICT:
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
  
  # calculate loss, for both TRAIN and EVAL modes
  loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)
  
  # configure the training Op for TRAIN mode
  if mode == tf.estimator.ModeKeys.TRAIN:
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
    train_op = optimizer.minimize(loss=loss,
                                 global_step=tf.train.get_global_step()) # ??
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)
  
  # add eval metrics for EVAL mode
  eval_metric_ops = {
      "accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions['classes'])
  }
  return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

# load training and test data
((train_data, train_labels), (eval_data, eval_labels)) = tf.keras.datasets.mnist.load_data()

train_data = train_data / np.float32(255)
train_labels = train_labels.astype(np.int32)

eval_data = eval_data / np.float32(255)
eval_labels = eval_labels.astype(np.int32)

# create teh estimator
mnist_estimator = tf.estimator.Estimator(model_fn=cnn_model_fn, model_dir="/tmp/mnist_convnet_model")

# set up logging hook
# set up logging for predictions
tensors_to_log = {"probabilities": "softmax_tensor"}
logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)

# train the model
train_input_fn = tf.estimator.inputs.numpy_input_fn(
                                      x={"x": train_data},
                                      y=train_labels, batch_size=100,
                                      num_epochs=None, 
                                      shuffle=True)

# train one step and display the probabilities
mnist_estimator.train(input_fn=train_input_fn, steps=1, hooks=[logging_hook])

# eval the model
eval_input_fn = tf.estimator.inputs.numpy_input_fn(x={"x": eval_data},
                                                  y=eval_labels,
                                                  num_epochs=1,
                                                  shuffle=False)
eval_results = mnist_estimator.evaluate(input_fn=eval_input_fn)
print(eval_results)
