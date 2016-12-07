from __future__ import print_function

import sys
import time
import math

# This is a placeholder for a Google-internal import.

import tensorflow as tf

from tensorflow.contrib.session_bundle import exporter
import archetype_input_data
tf.logging.set_verbosity(tf.logging.DEBUG)

tf.app.flags.DEFINE_integer('training_iteration', 1000,
                                    'number of training iterations.')
tf.app.flags.DEFINE_string('work_dir', '/myproject/', 'Working directory.')
tf.app.flags.DEFINE_integer('export_version', 1, 'version number of the model.')
FLAGS = tf.app.flags.FLAGS

def main(_):
  print('Training model...')
  archetype_data = archetype_input_data.read_data_sets(FLAGS.work_dir)
  sess = tf.InteractiveSession()
  serialized_tf_example = tf.placeholder(tf.string, name='tf_example')
  num_inputs = archetype_data.train.num_inputs
  num_classes = archetype_data.train.num_classes
  feature_configs = {
      'x': tf.FixedLenFeature(shape=[num_inputs], dtype=tf.float32),
  }
  tf_example = tf.parse_example(serialized_tf_example, feature_configs)
  x = tf.identity(tf_example['x'], name='x')  # use tf.identity() to assign name
  y_ = tf.placeholder('float', shape=[None, num_classes])
  w = tf.Variable(tf.zeros([num_inputs, num_classes]))
  b = tf.Variable(tf.zeros([num_classes]))
  sess.run(tf.initialize_all_variables())
  y = tf.cast(tf.nn.softmax(tf.matmul(x, w) + b, name='y'), tf.float32)
  cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
  train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
  values, indices = tf.nn.top_k(y, num_classes)
  prediction_classes = tf.contrib.lookup.index_to_string(
      tf.to_int64(indices), mapping=tf.constant([str(i) for i in range(num_classes)]))
  for _ in range(FLAGS.training_iteration):
    batch = archetype_data.train.next_batch(50)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
  print('training accuracy %g' %
        sess.run(accuracy,
                 feed_dict={x: archetype_data.test.decks,
                            y_: archetype_data.test.labels}))
  print('Done training!')

  # Export model
  # WARNING(break-tutorial-inline-code): The following code snippet is
  # in-lined in tutorials, please update tutorial documents accordingly
  # whenever code changes.
  export_path = sys.argv[-1]
  print('Exporting trained model to %s' % export_path)
  init_op = tf.group(tf.initialize_all_tables(), name='init_op')
  saver = tf.train.Saver(sharded=True)
  model_exporter = exporter.Exporter(saver)
  model_exporter.init(
      sess.graph.as_graph_def(),
      init_op=init_op,
      default_graph_signature=exporter.classification_signature(
          input_tensor=serialized_tf_example,
          classes_tensor=prediction_classes,
          scores_tensor=values),
      named_graph_signatures={
          'inputs': exporter.generic_signature({'decks': x}),
          'outputs': exporter.generic_signature({'scores': y})})
  model_exporter.export(export_path, tf.constant(FLAGS.export_version), sess)
  print('Done exporting!')

if __name__ == '__main__':
  tf.app.run()
