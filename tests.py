# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 18:43:46 2021

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

from  DataProcessing import realized_volatility, trainValidSplitNp, CutAndLinear, bookDataPreprocess, combineForTraining, npToDataset
from DataProcessing import RawDataInitialization, NpRepeatByCondition, NpSplitAndNormalized

from tensorboard.plugins.hparams import api as hp

from ML import *
import ML as ML
from  DataProcessing import *
import DataProcessing as DP
import pandas as pd
import numpy as np
import os
import glob
import plotly.express as px
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, MaxPool1D, BatchNormalization, Flatten, Dense,Dropout, Dropout, Input, Concatenate
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.callbacks import ModelCheckpoint,ReduceLROnPlateau, EarlyStopping
from tensorflow.keras import optimizers, regularizers
from tensorflow.keras.layers.experimental.preprocessing import Normalization
import time
import math 
import tensorflow_addons as tfa


from tensorboard.plugins.hparams import api as hp

"""
This file aimds to test training a single stock with 9 features, see the definition of the function DP.RawDataToBookAndTradeNp.
We follow the HyperParametersTuningScheme to tune our model
"""

data, labels = DP.RawDataToBookAndTradeNp(115)


"""
End the data processing part
"""

# =============================================================================
# train_book, train_file, train_trade = DP.RawDataInitialization([1,2])
# 
# data, labels = DP.RawDataToBookAndTradeNp(1, features=['log_return','log_return_trade'])
# =============================================================================

#Try to build a model without normalizing training data but only labels
train_data, train_labels, valid_data, valid_labels = DP.trainValidSplitNp(data, labels, 0.85)
mean = train_labels.mean()
std = train_labels.std()
train_labelsNormalized, valid_labelsNormalized = (train_labels-mean)/std, (valid_labels-mean)/std    
print("valid std = ", valid_labelsNormalized.std())

np.isnan(valid_data).sum()

trainDataset = tf.data.Dataset.from_tensor_slices((train_data, 
                                                   train_labelsNormalized)).batch(ML.BATCH_SIZE, drop_remainder=True).repeat(3)
validDataset = tf.data.Dataset.from_tensor_slices((valid_data, valid_labelsNormalized)).batch(ML.BATCH_SIZE, drop_remainder=True)

"""
The tuning routine
"""

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
HP_CONV_KERNEL_SIZE = hp.HParam("conv_kernel_size", hp.Discrete([3]))
HP_DENSE_LAYERS = hp.HParam("dense_layers", hp.IntInterval(1, 2))
HP_DROPOUT = hp.HParam("dropout", hp.RealInterval(0.35, 0.55))
HP_OPTIMIZER = hp.HParam("optimizer", hp.Discrete(["adam"])) # adagrad removed
HP_WEIGHT_DECAY = hp.HParam('weight_decay', hp.RealInterval(2e-5, 2e-3))
HP_INITIAL_LR = hp.HParam('initial_lr', hp.RealInterval(1e-4, 5e-3))
HP_CONV_UNITS = hp.HParam('conv_units', hp.Discrete([32,64]))
HP_LSTM_LAYERS = hp.HParam('lstm_layers', hp.Discrete([0,1,2]))
HP_DENSE_UNITS = hp.HParam('dense_units', hp.Discrete([64, 128]))
HP_LSTM_UNITS = hp.HParam('lstm_units', hp.Discrete([64, 128]))


HPARAMS = [
    HP_CONV_LAYERS,
    HP_CONV_KERNEL_SIZE,
    HP_DENSE_LAYERS,
    HP_DROPOUT, 
    HP_OPTIMIZER,
    HP_WEIGHT_DECAY,
    HP_INITIAL_LR,
    HP_CONV_UNITS,
    HP_LSTM_LAYERS,
    HP_DENSE_UNITS,
    HP_LSTM_UNITS
]

METRICS = [
    hp.Metric(
        "epoch_mean_squared_error",
        group="validation",
        display_name="mse (val.)",
    ),
    hp.Metric(
        "epoch_loss",
        group="validation",
        display_name="loss (val.)",
    ),
    hp.Metric(
        "epoch_mean_squared_error",
        group="train",
        display_name="mse (train)",
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
      initializer = tf.keras.initializers.lecun_uniform()     
      model = tf.keras.models.Sequential()
      model.add(Normalization())
      model.add(Conv1D(
          filters = 128,
          kernel_size = 3,
          padding = 'same',
          activation='relu',
          batch_input_shape = (ML.BATCH_SIZE,600,2),
          kernel_regularizer=regularizers.l1(hparams[HP_WEIGHT_DECAY]),
          # kernel_initializer = initializer
          ))
      model.add(BatchNormalization())
      conv_filers = hparams[HP_CONV_UNITS]
      for _ in xrange(hparams[HP_CONV_LAYERS]):
          
          model.add(Conv1D(
          filters = conv_filers,
          kernel_size = hparams[HP_CONV_KERNEL_SIZE],
          padding = 'same',  
          activation='relu',
          kernel_regularizer=regularizers.l1(hparams[HP_WEIGHT_DECAY]),
          # kernel_initializer = initializer
          ))
          
          model.add(MaxPool1D(pool_size = 2))
          
          conv_filers *= 2
          model.add(tf.keras.layers.Dropout(hparams[HP_DROPOUT], seed=rng.random()))
          
      for _ in xrange(hparams[HP_LSTM_LAYERS]): 
          
          model.add(tf.keras.layers.LSTM(hparams[HP_LSTM_UNITS], 
                                         stateful = True, 
                                         return_sequences = True, 
                                         # kernel_regularizer=regularizers.l1(hparams[HP_WEIGHT_DECAY]),
                                         kernel_initializer=initializer
                                         ))
          
          model.add(tf.keras.layers.Dropout(hparams[HP_DROPOUT], seed=rng.random()))
          
          model.add(BatchNormalization())

      
      model.add(tf.keras.layers.Flatten())

      dense_neurons = hparams[HP_DENSE_UNITS]
      for _ in xrange(hparams[HP_DENSE_LAYERS]):
          
          model.add(Dense(dense_neurons,  
                          activation='relu', 
                          # kernel_regularizer=regularizers.l1(hparams[HP_WEIGHT_DECAY]),
                           kernel_initializer=initializer
                          ))
          
          dense_neurons /= 2
          
          model.add(tf.keras.layers.Dropout(hparams[HP_DROPOUT], seed=rng.random()))
          
      model.add(Dense(1, activation='sigmoid'))
      
      adamOptimizer = tf.keras.optimizers.Adam(hparams[HP_INITIAL_LR])
      
      model.compile(loss = 'mean_squared_error', 
                optimizer = adamOptimizer,
                metrics = [tf.keras.metrics.MeanSquaredError()])
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
      update_freq=1,
      profile_batch=0)  # workaround for issue #2084
  check_point, early_stopping, learning_rate_reduction = ML.get_callbacks(base_logdir+"/weights/"+str(session_id),
                                                                          lr_reduction_factor=0.1,
                                                                          lr_reduction_patience=15,
                                                                          monitor='val_mean_squared_error',
                                                                          mode='min')
  hparams_callback = hp.KerasCallback(logdir, hparams)
  result = model.fit(
      trainDataset,
      validation_data = validDataset,
      # epochs=flags.FLAGS.num_epochs,
      epochs = 200,
      callbacks=[callback, hparams_callback, early_stopping],
      verbose=1
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

  num_of_sessions = 3
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
  np.random.seed(3)
  logdir = "logs/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
  shutil.rmtree(logdir, ignore_errors=True)
  print("Saving output to %s." % logdir)
  run_all(logdir=logdir, verbose=True)
  print("Done. Output saved to %s." % logdir)


if __name__ == "__main__":
  app.run(main)
  
  
"""compare out result with linear regression result"""
for stock in model_m

















         