import tensorflow as tf
import time

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
      self.hidden1 = tf.mul(tf.Variable(tf.random_normal([inputs_with_bias, hidden1], 0, 1.0)), init_factor_1)
      #@output  = (NMatrix.random(outdim)   - NMatrix.ones(outdim) /2) *@init_factor_2
      self.output_shape = [hidden1+1, outs]
      self.output = tf.mul(tf.Variable(tf.random_normal(self.output_shape, 0, 1.0)), init_factor_2)
    else:
      self.output= tf.Variable(tf.random_normal([inputs, outs], 0, 1.0))    
    
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
      
      self.output = tf.add(tf.mul(tf.transpose(tf.matmul(outdelta, a2_with_bias)), self.eta), self.output)
      self.hidden1 = tf.add(tf.mul(tf.transpose(tf.matmul(hiddelta1, input_with_bias)), self.eta * 0.1), self.hidden1)
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
    start_time = time.time()
    with tf.Session() as sess:
      sess.run(tf.initialize_all_variables())
      self.output = sess.run(self.output)
      self.hidden1 = sess.run(self.hidden1)
    print("--- %s seconds ---" % (time.time() - start_time))

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
                                
  def evaluate_for(self, input):
    input = tf.constant([input])
    #tf.initialize_all_variables()
    if self.hidden1 is not None:
      input_with_bias = self.add_bias(input, 1)
      outhid1 = tf.matmul(input_with_bias, self.hidden1)
      a2 = self.hid1func(outhid1)

      a2_with_bias = self.add_bias(a2, 1)
      # a2_with_bias = a2
      return self.outfunc(tf.matmul(a2_with_bias, self.output))
    else:
      return self.outfunc( tf.matmul(input, self.output))
    
  def extract_class_from(self, result):
    start_time = time.time()
    with tf.Session() as sess:
      sess.run(tf.initialize_all_variables())
      res = self.classes[sess.run(tf.argmax(tf.transpose(result), 0))]
    
    
    print("--- %s seconds ---" % (time.time() - start_time))
    return res

  #def close(self):
    #self.sess.close()
