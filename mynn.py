import tensorflow as tf

class MyNeuralNet:
  def __init__(self, outfunc=lambda x: x):
    self.outfunc = outfunc
    self.eta = tf.constant(1.0)
    self.sess = tf.InteractiveSession()
    
  def init_weights(self, inputs= 1, hidden1= None, hidden2= None, outs= 1):
    print(inputs);
    self.output= tf.Variable(tf.random_normal([inputs, outs], 0, 1.0))
    
  def train(self, input, t):
    input = tf.constant([input])
    t = tf.constant(t)
    tf.initialize_all_variables().run()
    out =self.outfunc( tf.matmul(input, self.output))
    #print(out.eval());
    e = tf.sub(t, out)
    delta = tf.transpose(e)
    diff = tf.matmul(delta,input)
    correction = tf.transpose(tf.mul(self.eta,diff))
    tf.add(self.output, correction)
    return e

  def set_classes(self, classes):
    self.classes = classes

  def evaluate_for(self, input):
    input = tf.constant(input)
    tf.initialize_all_variables().run()
    return self.outfunc( tf.mul(input, self.output))
    
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
