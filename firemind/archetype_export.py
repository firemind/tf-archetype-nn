from __future__ import print_function

import sys
import time

# This is a placeholder for a Google-internal import.

import tensorflow as tf

from tensorflow.contrib.session_bundle import exporter
from tensorflow_serving.firemind import archetype_input_data

tf.app.flags.DEFINE_string('work_dir', '/myproject/', 'Working directory.')
tf.app.flags.DEFINE_integer('export_version', 1, 'version number of the model.')
FLAGS = tf.app.flags.FLAGS

def inference(decks, input_size, hidden1_units, hidden2_units):
  """Build the MNIST model up to where it may be used for inference.
  Args:
    decks: Deck placeholder, from inputs().
    hidden1_units: Size of the first hidden layer.
    hidden2_units: Size of the second hidden layer.
  Returns:
    softmax_linear: Output tensor with the computed logits.
  """
  # Hidden 1
  with tf.name_scope('hidden1'):
    weights = tf.Variable(
        tf.truncated_normal([input_size, hidden1_units],
                            stddev=1.0 / math.sqrt(float(input_size))),
        name='weights')
    biases = tf.Variable(tf.zeros([hidden1_units]),
                         name='biases')
    hidden1 = tf.nn.relu(tf.matmul(decks, weights) + biases)
  # Hidden 2
  with tf.name_scope('hidden2'):
    weights = tf.Variable(
        tf.truncated_normal([hidden1_units, hidden2_units],
                            stddev=1.0 / math.sqrt(float(hidden1_units))),
        name='weights')
    biases = tf.Variable(tf.zeros([hidden2_units]),
                         name='biases')
    hidden2 = tf.nn.relu(tf.matmul(hidden1, weights) + biases)
  # Linear
  with tf.name_scope('softmax_linear'):
    weights = tf.Variable(
        tf.truncated_normal([hidden2_units, NUM_CLASSES],
                            stddev=1.0 / math.sqrt(float(hidden2_units))),
        name='weights')
    biases = tf.Variable(tf.zeros([NUM_CLASSES]),
                         name='biases')
    logits = tf.matmul(hidden2, weights) + biases
  return logits

def loss(logits, labels):
  """Calculates the loss from the logits and the labels.
  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size].
  Returns:
    loss: Loss tensor of type float.
  """
  labels = tf.to_int64(labels)
  cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
      logits, labels, name='xentropy')
  loss = tf.reduce_mean(cross_entropy, name='xentropy_mean')
  return loss


def training(loss, learning_rate):
  """Sets up the training Ops.
  Creates a summarizer to track the loss over time in TensorBoard.
  Creates an optimizer and applies the gradients to all trainable variables.
  The Op returned by this function is what must be passed to the
  `sess.run()` call to cause the model to train.
  Args:
    loss: Loss tensor, from loss().
    learning_rate: The learning rate to use for gradient descent.
  Returns:
    train_op: The Op for training.
  """
  # Add a scalar summary for the snapshot loss.
  tf.scalar_summary(loss.op.name, loss)
  # Create the gradient descent optimizer with the given learning rate.
  optimizer = tf.train.GradientDescentOptimizer(learning_rate)
  # Create a variable to track the global step.
  global_step = tf.Variable(0, name='global_step', trainable=False)
  # Use the optimizer to apply the gradients that minimize the loss
  # (and also increment the global step counter) as a single training step.
  train_op = optimizer.minimize(loss, global_step=global_step)
  return train_op

def evaluation(logits, labels):
  """Evaluate the quality of the logits at predicting the label.
  Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size], with values in the
      range [0, NUM_CLASSES).
  Returns:
    A scalar int32 tensor with the number of examples (out of batch_size)
    that were predicted correctly.
  """
  # For a classifier model, we can use the in_top_k Op.
  # It returns a bool tensor with shape [batch_size] that is true for
  # the examples where the label is in the top k (here k=1)
  # of all logits for that example.
  correct = tf.nn.in_top_k(logits, labels, 1)
  # Return the number of true entries.
  return tf.reduce_sum(tf.cast(correct, tf.int32))

def fill_feed_dict(data_set, decks_pl, labels_pl):
  """Fills the feed_dict for training the given step.
  A feed_dict takes the form of:
  feed_dict = {
      <placeholder>: <tensor of values to be passed for placeholder>,
      ....
  }
  Args:
    data_set: The set of images and labels, from input_data.read_data_sets()
    images_pl: The images placeholder, from placeholder_inputs().
    labels_pl: The labels placeholder, from placeholder_inputs().
  Returns:
    feed_dict: The feed dictionary mapping from placeholders to values.
  """
  # Create the feed_dict for the placeholders filled with the next
  # `batch size ` examples.
  decks_feed, labels_feed = data_set.next_batch(batch_size)
  feed_dict = {
      decks_pl: decks_feed,
      labels_pl: labels_feed,
  }
  return feed_dict

def do_eval(sess,
            eval_correct,
            images_placeholder,
            labels_placeholder,
            data_set):
  """Runs one evaluation against the full epoch of data.
  Args:
    sess: The session in which the model has been trained.
    eval_correct: The Tensor that returns the number of correct predictions.
    images_placeholder: The images placeholder.
    labels_placeholder: The labels placeholder.
    data_set: The set of images and labels to evaluate, from
      input_data.read_data_sets().
  """
  # And run one epoch of eval.
  true_count = 0  # Counts the number of correct predictions.
  steps_per_epoch = data_set.num_examples // batch_size
  num_examples = steps_per_epoch * batch_size
  for step in xrange(steps_per_epoch):
    feed_dict = fill_feed_dict(data_set,
                               images_placeholder,
                               labels_placeholder)
    true_count += sess.run(eval_correct, feed_dict=feed_dict)
  precision = true_count*1.0 / num_examples
  print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
        (num_examples, true_count, precision))
 
 
batch_size = 160
learning_rate = 0.05
max_steps = 3000
train_dir = './mytrain'
with tf.Graph().as_default():
  train_data_set = archetype_input_data.read_data_sets(FLAGS.work_dir).train
  decks_placeholder = tf.placeholder(tf.float32, shape=(batch_size, train_data_set.num_inputs))
  labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
  logits = inference(decks_placeholder,
                     train_data_set.num_inputs,
                           64,
                           16)
  loss_op = loss(logits, labels_placeholder)

  # Add to the Graph the Ops that calculate and apply gradients.
  train_op = training(loss_op, learning_rate)

  # Add the Op to compare the logits to the labels during evaluation.
  eval_correct = evaluation(logits, labels_placeholder)

  # Build the summary operation based on the TF collection of Summaries.
  summary_op = tf.merge_all_summaries()

  # Create a saver for writing training checkpoints.
  saver = tf.train.Saver()

  # Create a session for running Ops on the Graph.
  sess = tf.Session()

  # Run the Op to initialize the variables.
  init = tf.initialize_all_variables()
  sess.run(init)

  # Instantiate a SummaryWriter to output summaries and the Graph.
  summary_writer = tf.train.SummaryWriter(train_dir, sess.graph)

  # And then after everything is built, start the training loop.
  for step in xrange(max_steps):
    start_time = time.time()

    # Fill a feed dictionary with the actual set of images and labels
    # for this particular training step.
    feed_dict = fill_feed_dict(train,
                               decks_placeholder,
                               labels_placeholder)


    # Run one step of the model.  The return values are the activations
    # from the `train_op` (which is discarded) and the `loss` Op.  To
    # inspect the values of your Ops or variables, you may include them
    # in the list passed to sess.run() and the value tensors will be
    # returned in the tuple from the call.
    _, loss_value = sess.run([train_op, loss_op],
                             feed_dict=feed_dict)

    duration = time.time() - start_time

    # Write the summaries and print an overview fairly often.
    if step % 100 == 0:
      # Print status to stdout.
      print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
      # Update the events file.
      summary_str = sess.run(summary_op, feed_dict=feed_dict)
      summary_writer.add_summary(summary_str, step)
      summary_writer.flush()

    # Save a checkpoint and evaluate the model periodically.
    if (step + 1) % 1000 == 0 or (step + 1) == max_steps:
      #saver.save(sess, FLAGS.train_dir, global_step=step)
      # Evaluate against the training set.
      print('Training Data Eval:')
      do_eval(sess,
              eval_correct,
              decks_placeholder,
              labels_placeholder,
              train)
      # Evaluate against the validation set.
      #print('Validation Data Eval:')
      #do_eval(sess,
      #        eval_correct,
      #        decks_placeholder,
      #        labels_placeholder,
      #        data_sets.validation)
      # Evaluate against the test set.
      print('Test Data Eval:')
      do_eval(sess,
              eval_correct,
              decks_placeholder,
              labels_placeholder,
              test)

  export_path = "./archetype-model"
  saver = tf.train.Saver(sharded=True)
  model_exporter = exporter.Exporter(saver)
  model_exporter.init(
    sess.graph.as_graph_def(),
    named_graph_signatures={
      'inputs': exporter.generic_signature({'decks': decks_placeholder}),
      'outputs': exporter.generic_signature({'classes': labels_placeholder})})
  model_exporter.export(export_path, tf.constant(FLAGS.export_version), sess)
