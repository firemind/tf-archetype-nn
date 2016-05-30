import tensorflow as tf

class MyNeuralNet:
  def __init__(self, outfunc=lambda x: x,  hid1func=lambda x: x, deriv1func= lambda x: x):
    self.outfunc = outfunc
    self.hid1func = hid1func
    self.deriv1func = deriv1func
    self.eta = tf.constant(1.0)
    self.sess = tf.InteractiveSession()
    
  def init_weights(self, inputs= 1, hidden1= None, hidden2= None, outs= 1):
    if hidden1:
      inputs_with_bias = inputs # + 1
      init_factor_1 = 0.01 / ((inputs_with_bias) ** (0.5))
      init_factor_2 = 0.01 / (hidden1 ** (0.5))      
      #@hidden1 = (NMatrix.random(hid1dim) - NMatrix.ones(hid1dim) /2)*@init_factor_1
      self.hidden1 = tf.mul(tf.Variable(tf.random_normal([inputs_with_bias, hidden1], 0, 1.0)), init_factor_1)
      #@output  = (NMatrix.random(outdim)   - NMatrix.ones(outdim) /2) *@init_factor_2
      self.output = tf.mul(tf.Variable(tf.random_normal([hidden1, outs], 0, 1.0)), init_factor_2)

    else:
      self.output= tf.Variable(tf.random_normal([inputs, outs], 0, 1.0))    
    
  def train(self, input, t):
    input = tf.constant([input])
    t = tf.constant(t)
    tf.initialize_all_variables().run()
    if self.hidden1 is not None:
 
      # outhid1 = input.dot @hidden1
      outhid1 = tf.matmul(input, self.hidden1)
      a2 = self.hid1func(outhid1)

      #a2_with_bias = add_bias(a2)
      a2_with_bias = a2
       
      out = tf.matmul(a2_with_bias, self.output)
      a3 = self.outfunc( out )
      e = tf.sub(t,a3)
      outdelta = tf.transpose(e)
      hiddelta1 = self.calc_hid_delta(self.deriv1func(tf.transpose(outhid1)), self.output, outdelta)
      
      self.output  += tf.mul(tf.transpose(tf.matmul(outdelta, a2_with_bias)), self.eta)
      
      self.hidden1 += tf.mul(tf.transpose(tf.matmul(hiddelta1, input)), self.eta * 0.1)
    else:
        
        out =self.outfunc( tf.matmul(input, self.output))
        #print(out.eval());
        e = tf.sub(t, out)
        delta = tf.transpose(e)
        diff = tf.matmul(delta,input)
        correction = tf.transpose(tf.mul(self.eta,diff))
        tf.add(self.output, correction)
        return e

  def calc_hid_delta(self, outhid, output, outdelta):
    m = tf.matmul(output, outdelta)
    #m = remove_bias(m)
    return tf.mul(m, outhid)
    
  def set_classes(self, classes):
    self.classes = classes

  #def add_bias(self, m):
  #  num_hid_nodes = hidden1.shape[1]
  #  m_with_bias = NMatrix.zeroes([1,num_hid_nodes+1])
  #  m_with_bias[0,0..num_hid_nodes] = m
  #  m_with_bias[0,num_hid_nodes] = 1.0 
  #  m
  
  def evaluate_for(self, input):
    input = tf.constant([input])
    tf.initialize_all_variables().run()
    return self.outfunc( tf.matmul(input, self.output))
    
  def extract_class_from(self, result):
    max_index = None
    max_val = -100000
    for index, val in enumerate(result):
      if val > max_val:
        max_index = index
        max_val = val
    return max_index and [self.classes[max_index],max_val]

  def close(self):
    self.sess.close()
