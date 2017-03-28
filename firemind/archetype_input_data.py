# Copyright 2016 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

#!/usr/bin/env python2.7

"""Functions for downloading and reading MNIST data."""

from __future__ import print_function

import tensorflow as tf
import gzip
import os
import math
import csv
import random

import numpy
from six.moves import urllib

def dense_to_one_hot(labels_dense, num_classes=10):
  """Convert class labels from scalars to one-hot vectors."""
  num_labels = labels_dense.shape[0]
  index_offset = numpy.arange(num_labels) * num_classes
  labels_one_hot = numpy.zeros((num_labels, num_classes))
  labels_one_hot.flat[index_offset + labels_dense.ravel()] = 1
  return labels_one_hot

class DataSet(object):

  def __init__(self, decks, labels, classes, dtype=tf.float32):
    dtype = tf.as_dtype(dtype).base_dtype
    if dtype not in (tf.uint8, tf.float32):
      raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                      dtype)


    self._num_examples = len(decks)
    self._num_inputs = len(decks[0])
    self._classes = classes

    # Convert shape from [num examples, rows, columns, depth]
    # to [num examples, rows*columns] (assuming depth == 1)

    self._decks = numpy.array(decks).astype(numpy.float32)
    self._labels = dense_to_one_hot(numpy.array(labels), len(classes)) #.astype(numpy.float32)
    self._epochs_completed = 0
    self._index_in_epoch = 0

  @property
  def decks(self):
    return self._decks

  @property
  def labels(self):
    return self._labels

  @property
  def num_inputs(self):
    return self._num_inputs

  @property
  def num_examples(self):
    return self._num_examples

  @property
  def num_classes(self):
    return len(self._classes)

  @property
  def classes(self):
    return self._classes

  @property
  def epochs_completed(self):
    return self._epochs_completed

  def next_batch(self, batch_size):
    """Return the next `batch_size` examples from this data set."""

    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = numpy.arange(self._num_examples)
      numpy.random.shuffle(perm)
      self._decks = self._decks[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._decks[start:end], self._labels[start:end]

def read_data_sets(train_dir):
  """Return training, validation and testing data sets."""

  class DataSets(object):
    pass

  data_sets = DataSets()

  archetypes = []
  header = None
  cards = None
  classes = set()
  input_size = None
  with open(train_dir+'archetypes.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',', quotechar='"')
    for row in reader:
      if header == None:
        header = row
        cards = map(float, row[3:])
        input_size = len(row)-3
      #elif row[2] == "37":
      else:
        c = float(row[1])
        classes.add(c)
        rec = [map(float, row[3:]), c]
        archetypes.append(rec)
      #else:
        #print row[2];
        #break

  decks = []
  labels = []
  classes = list(classes)
  max_samples = 8000
  random.shuffle(archetypes)
  for rec in archetypes[:max_samples]:
      decks.append(rec[0])
      #labels.append( map(lambda x: (1.0 if x == rec[1] else 0.0), classes))
      labels.append(classes.index(rec[1]))


  split = int(len(decks) * 0.8)
  tv_split = int(split* 0.8)

  data_sets.train = DataSet(decks[(split):], labels[(split):], classes)
  data_sets.validation = DataSet(decks[tv_split:split],     labels[tv_split:split], classes)
  data_sets.test = DataSet(decks[:tv_split],     labels[:tv_split], classes)

  return data_sets
