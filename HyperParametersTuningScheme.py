# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 01:42:50 2021

@author: taoso
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os.path
import random
import shutil
import datetime

import ML as ML
from absl import app
from absl import flags
import numpy as np
import six
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, MaxPool1D, BatchNormalization, Flatten, Dense,Dropout, Dropout, Input, Concatenate
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import ModelCheckpoint,ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import optimizers, regularizers
from sklearn.utils import shuffle

from  DataProcessing import dataToPd, realized_volatility, trainValidSplitNp, CutAndLinear, bookDataPreprocess, combineForTraining, npToDataset
from DataProcessing import RawDataInitialization, NpRepeatByCondition, NpSplitAndNormalized

from tensorboard.plugins.hparams import api as hp


train_book, train_file, train_trade = RawDataInitialization([115])
stock_id = 'stock_id=115'
interval = 20
dfWithPrediction = CutAndLinear(train_book, train_file, stock_id, interval)

train_target = dfWithPrediction[['stock_id','time_id','abnormal']]
train_target = train_target.rename({'abnormal':'target'}, axis = 1)
train_target['target'] = train_target['target'].astype('uint8')

# =============================================================================
# store it into newLabelsFromLinearModel_stock_id=43.csv 
# =============================================================================

train_target.to_csv("newLabelsFromLinearModel_stock_id=43.csv")
train_np, labels_np = bookDataPreprocess(train_book[stock_id], train_target, stock_id)

trainNpRepeated, labelsNpRepeated = NpRepeatByCondition(train_np, labels_np, 1, times_repeat=2)
trainNpRepeatedShuffled, labelsNpRepeatedShuffled = shuffle(trainNpRepeated, labelsNpRepeated)

trainBookAndLabel, validationBookAndLabel, normalizers = NpSplitAndNormalized(trainNpRepeatedShuffled, 
                                                                              labelsNpRepeatedShuffled, 0.8, categoricalLabels=True)
trainDataset, validDataset = npToDataset(trainBookAndLabel[0], 
                                         trainBookAndLabel[1], validationBookAndLabel[0], validationBookAndLabel[1])



flags.DEFINE_integer(
    "num_session_groups",
    100,
    "The approximate number of session groups to create.",
)
flags.DEFINE_string(
    "logdir",
    "/tmp/hparams_demo",
    "The directory to write the summary information to.",
)
flags.DEFINE_integer(
    "summary_freq",
    600,
    "Summaries will be written every n steps, where n is the value of "
        "this flag.",
)
flags.DEFINE_integer(
    "num_epochs",
    5,
    "Number of epochs per trial.",
)


HP_CONV_LAYERS = hp.HParam("conv_layers", hp.IntInterval(1, 2))
HP_CONV_KERNEL_SIZE = hp.HParam("conv_kernel_size", hp.Discrete([3, 5]))
HP_DENSE_LAYERS = hp.HParam("dense_layers", hp.IntInterval(1, 1))
HP_DROPOUT = hp.HParam("dropout", hp.RealInterval(0.1, 0.6))
HP_OPTIMIZER = hp.HParam("optimizer", hp.Discrete(["adam"])) # adagrad removed
HP_WEIGHT_DECAY = hp.HParam('weight_decay', hp.RealInterval(0.00005, 0.00015))
HP_INITIAL_LR = hp.HParam('initial_lr', hp.Discrete([0.1, 0.01, 0.001, 0.0001]))
HP_CONV_UNITS = hp.HParam('dense_units', hp.Discrete([32,64,96]))

HPARAMS = [
    HP_CONV_LAYERS,
    HP_CONV_KERNEL_SIZE,
    HP_DENSE_LAYERS,
    HP_DROPOUT,
    HP_OPTIMIZER,
    HP_WEIGHT_DECAY,
    HP_INITIAL_LR,
    HP_CONV_UNITS
]

METRICS = [
    hp.Metric(
        "epoch_accuracy",
        group="validation",
        display_name="accuracy (val.)",
    ),
    hp.Metric(
        "epoch_loss",
        group="validation",
        display_name="loss (val.)",
    ),
    hp.Metric(
        "batch_accuracy",
        group="train",
        display_name="accuracy (train)",
    ),
    hp.Metric(
        "batch_loss",
        group="train",
        display_name="loss (train)",
    ),
]


def modelAbnormalityClassifier(hparams, seed):
      """Create a Keras model with the given hyperparameters.
      Args:
      hparams: A dict mapping hyperparameters in `HPARAMS` to values.
      seed: A hashable object to be used as a random seed (e.g., to
      construct dropout layers in the model).
      Returns:
      A compiled Keras model.
      """
      rng = random.Random(seed)
      
      model = tf.keras.models.Sequential()
      model.add(Conv1D(
          filters = hparams[HP_CONV_UNITS],
          kernel_size = hparams[HP_CONV_KERNEL_SIZE],
          padding = 'same',
          activation='relu',
          batch_input_shape = (ML.BATCH_SIZE,600,1)))
      conv_filers = 32
      for _ in xrange(hparams[HP_CONV_LAYERS]):
          model.add(Conv1D(
          filters = conv_filers,
          kernel_size = hparams[HP_CONV_KERNEL_SIZE],
          padding = 'same',
          activation='relu',
          ))
          
          model.add(MaxPool1D(pool_size = 2, padding = 'same'))
          conv_filers *= 2
          model.add(tf.keras.layers.Dropout(hparams[HP_DROPOUT], seed=rng.random()))
      
      model.add(tf.keras.layers.LSTM(64, 
                                     stateful = True, 
                                     return_sequences = True, 
                                     kernel_regularizer=regularizers.l1(hparams[HP_WEIGHT_DECAY])
                                     ))
      
      model.add(tf.keras.layers.Flatten())
      model.add(tf.keras.layers.Dropout(hparams[HP_DROPOUT], seed=rng.random()))
      
      dense_neurons = 64
      for _ in xrange(hparams[HP_DENSE_LAYERS]):
          model.add(Dense(dense_neurons,  
                          activation='relu', 
                          kernel_regularizer=regularizers.l1(hparams[HP_WEIGHT_DECAY])
                          ))
          dense_neurons /= 2
          model.add(tf.keras.layers.Dropout(hparams[HP_DROPOUT], seed=rng.random()))
          
      model.add(Dense(2, activation = 'softmax'))
      
      adamOptimizer = tf.keras.optimizers.Adam(hparams[HP_INITIAL_LR])
      
      model.compile(loss = 'sparse_categorical_crossentropy', 
                optimizer = adamOptimizer,
                metrics = ['accuracy'])
      return model
      
  
        
        


def run(trainDataset, validDataset, base_logdir, session_id, hparams):
  """Run a training/validation session.
  Flags must have been parsed for this function to behave.
  Args:
    data: The data as loaded by `prepare_data()`.
    base_logdir: The top-level logdir to which to write summary data.
    session_id: A unique string ID for this session.
    hparams: A dict mapping hyperparameters in `HPARAMS` to values.
  """
  model = modelAbnormalityClassifier(hparams=hparams, seed=session_id)
  logdir = os.path.join(base_logdir, session_id)

  callback = tf.keras.callbacks.TensorBoard(
      logdir,
      update_freq=flags.FLAGS.summary_freq,
      profile_batch=0)  # workaround for issue #2084
  check_point, early_stopping, learning_rate_reduction = ML.get_callbacks(base_logdir+"/weights/"+str(session_id))
  hparams_callback = hp.KerasCallback(logdir, hparams)
  result = model.fit(
      trainDataset,
      validation_data = validDataset,
      # epochs=flags.FLAGS.num_epochs,
      epochs = 1000,
      shuffle=False,
      callbacks=[callback, hparams_callback, learning_rate_reduction, early_stopping],
      verbose=0
  )




def run_all(logdir, verbose=False):
  """Perform random search over the hyperparameter space.
  Arguments:
    logdir: The top-level directory into which to write data. This
      directory should be empty or nonexistent.
    verbose: If true, print out each run's name as it begins.
  """
  rng = random.Random(0)

  with tf.summary.create_file_writer(logdir).as_default():
    hp.hparams_config(hparams=HPARAMS, metrics=METRICS)

  num_of_sessions = 40
  sessions_per_group = 1
  num_sessions = int(num_of_sessions * sessions_per_group)
  session_index = 0  # across all session groups
  for group_index in xrange(num_of_sessions):
    hparams = {h: h.domain.sample_uniform(rng) for h in HPARAMS}
    hparams_string = str(hparams)
    for repeat_index in xrange(sessions_per_group):
      session_id = str(session_index)
      session_index += 1
      if verbose:
        print(
            "--- Running training session %d/%d"
            % (session_index, num_sessions)
        )
        print(hparams_string)
        print("--- repeat #: %d" % (repeat_index + 1))
      run(
          trainDataset = trainDataset,
          validDataset = validDataset,
          base_logdir=logdir,
          session_id=session_id,
          hparams=hparams,
      )


def main(unused_argv):
  np.random.seed(0)
  logdir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  shutil.rmtree(logdir, ignore_errors=True)
  print("Saving output to %s." % logdir)
  run_all(logdir=logdir, verbose=True)
  print("Done. Output saved to %s." % logdir)


if __name__ == "__main__":
  app.run(main)
  
  