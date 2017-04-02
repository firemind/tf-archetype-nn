from keras.models import Sequential

from keras.layers import Dense, Activation, Embedding, Flatten, Lambda, LSTM, Reshape, Layer, Conv1D, MaxPooling1D, Input
import archetype_input_data
import tensorflow as tf
from tensorflow.contrib.session_bundle import exporter
from keras import backend as K
from keras.utils import np_utils

sess = tf.Session()
K.set_session(sess)


tf.app.flags.DEFINE_integer('training_iteration', 1000, 'number of training iterations.')
FLAGS = tf.app.flags.FLAGS

ad = archetype_input_data.read_data_sets('/archetype-data/')
num_inputs = ad.train.num_inputs
num_classes = ad.train.num_classes

model = Sequential()
model.add(Dense(256, input_dim=num_inputs))
model.add(Activation('relu'))
model.add(Dense(256, input_dim=num_inputs))
model.add(Activation('relu'))
model.add(Dense(len(ad.train.classes)))
model.add(Activation('softmax'))


model.compile(loss='categorical_crossentropy',
        optimizer='sgd',
        metrics=['accuracy'])


#for _ in range(FLAGS.training_iteration):
  #batch = ad.train.next_batch(1)
  #print batch
  #model.train_on_batch(batch[0][0], batch[1][0])
print len(ad.train.decks)
print ad.train.decks[0]
print ad.train.labels[0]
model.fit(ad.train.decks, ad.train.labels, epochs=5, batch_size=32)

loss_and_metrics = model.evaluate(ad.test.decks, ad.test.labels, batch_size=128)
print(loss_and_metrics)

saver = tf.train.Saver(sharded=True)
model_exporter = exporter.Exporter(saver)
signature = exporter.classification_signature(input_tensor=model.input, scores_tensor=model.output)
model_exporter.init(sess.graph.as_graph_def(), default_graph_signature=signature)
model_dir = '/archetype-data/model'
model_version = 1
model_exporter.export(model_dir, tf.constant(model_version), sess)

