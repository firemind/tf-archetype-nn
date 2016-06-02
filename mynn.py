import tensorflow as tf

class MyNeuralNet:
  def __init__(self, outfunc=lambda x: x,  hid1func=lambda x: x, deriv1func= lambda x: x):
    self.outfunc = outfunc
    self.hid1func = hid1func
    self.deriv1func = deriv1func
    self.eta = tf.constant(1.0)
    #self.sess = tf.InteractiveSession()
    
  def init_weights(self, inputs= 1, hidden1= None, hidden2= None, outs= 1):
    if hidden1:
      self.input_size = inputs
      inputs_with_bias = inputs  + 1
      init_factor_1 = 0.01 / ((inputs_with_bias) ** (0.5))
      init_factor_2 = 0.01 / (hidden1 ** (0.5))      
      #@hidden1 = (NMatrix.random(hid1dim) - NMatrix.ones(hid1dim) /2)*@init_factor_1
      self.hidden1_size =  hidden1
      self.output_shape = [hidden1+1, outs]
      with tf.Session() as sess:
        self.hidden1 = tf.Variable(tf.random_normal([inputs_with_bias, hidden1], 0, init_factor_1))
        self.output = tf.Variable(tf.random_normal(self.output_shape, 0, init_factor_2))
    else:
      self.output= tf.Variable(tf.random_normal([inputs, outs], 0, 1.0))    
  def inference(images, hidden1_units, hidden2_units):
      """Build the MNIST model up to where it may be used for inference.
      Args:
        images: Images placeholder, from inputs().
        hidden1_units: Size of the first hidden layer.
        hidden2_units: Size of the second hidden layer.
      Returns:
        softmax_linear: Output tensor with the computed logits.
      """
      # Hidden 1
      with tf.name_scope('hidden1'):
        weights = tf.Variable(
            tf.truncated_normal([IMAGE_PIXELS, hidden1_units],
                                stddev=1.0 / math.sqrt(float(IMAGE_PIXELS))),
            name='weights')
        biases = tf.Variable(tf.zeros([hidden1_units]),
                             name='biases')
        hidden1 = tf.nn.relu(tf.matmul(images, weights) + biases)
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

  def new_train(self, inputs_pl, labels_pl):
    optimizer = tf.train.GradientDescentOptimizer(self.eta)
    global_step = tf.Variable(0, name='global_step', trainable=False)
    # Use the optimizer to apply the gradients that minimize the loss
    # (and also increment the global step counter) as a single training step.
    train_op = optimizer.minimize(loss, global_step=global_step)
    return train_op
  def train(self, input, t):
    input = tf.constant([input])
    t = tf.constant(t)
    #tf.initialize_all_variables()
    if self.hidden1 is not None:
 
      # outhid1 = input.dot @hidden1
      input_with_bias = self.add_bias(input, 1)
      outhid1 = tf.matmul(input_with_bias, self.hidden1)
      a2 = self.hid1func(outhid1)

      a2_with_bias = self.add_bias(a2, 1)
      #a2_with_bias = a2
       
      out = tf.matmul(a2_with_bias, self.output)
      a3 = self.outfunc( out )
      e = tf.sub(t,a3)
      outdelta = tf.transpose(e)
      hiddelta1 = self.calc_hid_delta(self.deriv1func(tf.transpose(outhid1)), self.output, outdelta, self.hidden1_size)
      
      tf.assign(self.output, tf.add(tf.mul(tf.transpose(tf.matmul(outdelta, a2_with_bias)), self.eta), self.output))
      tf.assign(self.hidden1, tf.add(tf.mul(tf.transpose(tf.matmul(hiddelta1, input_with_bias)), self.eta * 0.1), self.hidden1))
    else:
        
        out =self.outfunc( tf.matmul(input, self.output))
        #print(out.eval());
        e = tf.sub(t, out)
        delta = tf.transpose(e)
        diff = tf.matmul(delta,input)
        correction = tf.transpose(tf.mul(self.eta,diff))
        tf.add(self.output, correction)
        return e

  def finish_training(self):
    with tf.Session() as sess:
      sess.run(tf.initialize_all_variables())
      self.output = sess.run(self.output)
      self.hidden1 = sess.run(self.hidden1)

  def calc_hid_delta(self, outhid, output, outdelta, size):
    m = tf.matmul(output, outdelta)
    m = self.remove_bias(m, size)
    return tf.mul(m, outhid)
    
  def set_classes(self, classes):
    self.classes = classes

  def add_bias(self, m, size):
    #num_hid_nodes = hidden1.shape[1]
    #m_with_bias = NMatrix.zeroes([1,num_hid_nodes+1])
    #m_with_bias[0,0..num_hid_nodes] = m
    #m_with_bias[0,num_hid_nodes] = 1.0
    ones_shape = [1, size]
    #print ones_shape
    the_ones = tf.ones(ones_shape)
    #print the_ones
    #print m
    m_with_bias = tf.concat(1, [m, the_ones])
    return m_with_bias
  
  def remove_bias(self, m, size):
    s = [size, 1]
    #print s
    return tf.slice(m, [0, 0], s) 
                                
  def evaluate_op(self, input_placeholder):
    if self.hidden1 is not None:
      input_with_bias = self.add_bias(input_placeholder, 1)
      outhid1 = tf.matmul(input_with_bias, self.hidden1)
      a2 = self.hid1func(outhid1)

      a2_with_bias = self.add_bias(a2, 1)
      # a2_with_bias = a2
      return self.outfunc(tf.matmul(a2_with_bias, self.output))
    else:
      return self.outfunc( tf.matmul(input, self.output))
    
  def extract_class_from(self, result, sess):
    sess.run(tf.initialize_all_variables())
    res = self.classes[sess.run(tf.argmax(tf.transpose(result), 0))]
    return res

  #def close(self):
    #self.sess.close()
