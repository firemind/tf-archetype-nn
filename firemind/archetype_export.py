from __future__ import print_function

import sys
import time
import math

# This is a placeholder for a Google-internal import.

import tensorflow as tf
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils
from tensorflow.python.util import compat
from tensorflow.contrib import lookup as lookup_lib


from tensorflow.contrib.session_bundle import exporter
import archetype_input_data
tf.logging.set_verbosity(tf.logging.DEBUG)

tf.app.flags.DEFINE_integer('training_iteration', 1000,
                                    'number of training iterations.')
tf.app.flags.DEFINE_string('work_dir', '/archetype-data/', 'Working directory.')
tf.app.flags.DEFINE_integer('export_version', 1, 'version number of the model.')
FLAGS = tf.app.flags.FLAGS

def multilayer_perceptron(x, weights, biases):
  layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
  layer_1 = tf.nn.sigmoid(layer_1)
  # Hidden layer with RELU activation
  #layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
  #layer_2 = tf.nn.sigmoid(layer_2)
  # Output layer with linear activation
  #out_layer = tf.nn.softmax(tf.matmul(layer_1, weights['out']) + biases['out'])
  out_layer = tf.matmul(layer_1, weights['out'])+ biases['out']
  return out_layer

def main(_):
  print('Training model...')
  archetype_data = archetype_input_data.read_data_sets(FLAGS.work_dir)
  sess = tf.InteractiveSession()
  serialized_tf_example = tf.placeholder(tf.string, name='tf_example')
  num_inputs = archetype_data.train.num_inputs
  num_classes = archetype_data.train.num_classes
  print("Inputs: ", num_inputs)
  print("Classes: ", num_classes)
  feature_configs = {
      'x': tf.FixedLenFeature(shape=[num_inputs], dtype=tf.float32),
  }
  tf_example = tf.parse_example(serialized_tf_example, feature_configs)
  x = tf.identity(tf_example['x'], name='x')  # use tf.identity() to assign name
  y_ = tf.placeholder('float', shape=[None, num_classes])
  w = tf.Variable(tf.zeros([num_inputs, num_classes]))
  b = tf.Variable(tf.zeros([num_classes]))
  n_hidden_1 = 40 # 1st layer number of features
  n_hidden_2 = 40 # 2nd layer number of features
  weights = {
    'h1': tf.Variable(tf.truncated_normal([num_inputs, n_hidden_1])),
    'h2': tf.Variable(tf.truncated_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.truncated_normal([n_hidden_2, num_classes]))
    #'out': tf.Variable(tf.zeros([num_inputs, num_classes]))
  }
  biases = {
    'b1': tf.Variable(tf.zeros([n_hidden_1])),
    'b2': tf.Variable(tf.zeros([n_hidden_2])),
    'out': tf.Variable(tf.zeros([num_classes]))
  }
  init = tf.global_variables_initializer()
  sess.run(init)
  y = multilayer_perceptron(x, weights, biases)
  #y= tf.cast(tf.nn.softmax(tf.matmul(x, w) + b, name='y'), tf.float32)
  cross_entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_)) #-tf.reduce_sum(y_ * tf.log(y))
  #cross_entropy = -tf.reduce_sum(y_ * tf.log(y))
  train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
  #train_step = tf.train.AdamOptimizer(learning_rate=0.01).minimize(cross_entropy)
  values, indices = tf.nn.top_k(y, num_classes)
  mapping_string = tf.constant([str(i) for i in archetype_data.train.classes])
  prediction_classes = tf.contrib.lookup.index_to_string(
    tf.to_int64(indices), mapping=mapping_string)
  #mapping_string = tf.constant([str(i) for i in range(num_classes)])
  #indices = tf.to_int64(indices)
  #mapping_string = tf.constant(["emerson", "lake", "palmer"])
  #table = lookup_lib.index_to_string_from_tensor(
  #  mapping_string, default_value="UNKNOWN")
  #prediction_classes = table.lookup(indices)
  #tf.tables_initializer().run()
  for _ in range(FLAGS.training_iteration):
    batch = archetype_data.train.next_batch(500)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})
  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
  print(archetype_data.test.decks[0])
  print(archetype_data.test.labels[0])
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
  builder = saved_model_builder.SavedModelBuilder(export_path)


  # Build the signature_def_map.
  classification_inputs = utils.build_tensor_info(serialized_tf_example)
  classification_outputs_classes = utils.build_tensor_info(prediction_classes)
  classification_outputs_scores = utils.build_tensor_info(values)

  classification_signature = signature_def_utils.build_signature_def(
      inputs={signature_constants.CLASSIFY_INPUTS: classification_inputs},
      outputs={
          signature_constants.CLASSIFY_OUTPUT_CLASSES:
              classification_outputs_classes,
          signature_constants.CLASSIFY_OUTPUT_SCORES:
              classification_outputs_scores
      },
      method_name=signature_constants.CLASSIFY_METHOD_NAME)

  tensor_info_x = utils.build_tensor_info(x)
  tensor_info_y = utils.build_tensor_info(y)

  prediction_signature = signature_def_utils.build_signature_def(
     inputs={'decks': tensor_info_x},
     outputs={'scores': tensor_info_y},
     method_name=signature_constants.PREDICT_METHOD_NAME)

  legacy_init_op = tf.group(tf.initialize_all_tables(), name='legacy_init_op')
  builder.add_meta_graph_and_variables(
      sess, [tag_constants.SERVING],
      signature_def_map={
          'predict_decks':
              prediction_signature,
          signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
              classification_signature,
      },
      legacy_init_op=legacy_init_op)

  builder.save()

  print('Done exporting!')

if __name__ == '__main__':
  tf.app.run()
