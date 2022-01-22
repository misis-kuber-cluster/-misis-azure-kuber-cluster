import numpy as np
import pandas as pd
import sys

sys.stderr = open('file', 'w')
import os

# Suppress tensorflow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"

from tensorflow import keras
from tensorflow.keras import backend as K
from tensorflow.keras.applications.resnet import ResNet50
from tensorflow.keras.optimizers import Adam
from nltk.corpus import stopwords
import matplotlib.pyplot as plt
import seaborn as sns
import os
import argparse
import shutil
from glob import glob
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from time import time
import cv2
import tensorflow as tf
from collections import Counter
import DatasetValidation
 import horovod.tensorflow.keras as hvd

# Initialize Horovod
hvd.init()

metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall(), tf.keras.metrics]

# Pin GPU to be used to process local rank (one GPU per process)
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
if gpus:
    tf.config.experimental.set_visible_devices(gpus[hvd.local_rank()], 'GPU')

# Parse input arguments
parser = argparse.ArgumentParser(description='Team 1 Kubernteset cluster with Horovod',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--log-dir', default='./logs',
                    help='tensorboard log directory')
parser.add_argument('--batch-size', type=int, default=16,
                    help='input batch size for training')
parser.add_argument('--epochs', type=int, default=5,
                    help='number of epochs to train')
parser.add_argument('--base-lr', type=float, default=0.01,
                    help='learning rate for a single GPU')
parser.add_argument('--momentum', type=float, default=0.9,
                    help='SGD momentum')
parser.add_argument('--wd', type=float, default=0.000005,
                    help='weight decay')
parser.add_argument('--target-accuracy', type=float, default=.85,
                    help='Target accuracy to stop training')
parser.add_argument('--patience', type=float, default=2,
                    help='Number of epochs that meet target before stopping')
parser.add_argument('--use-checkpointing', default=False, action='store_true')
# TODO: Step 9: register `--warmup-epochs`
parser.add_argument('--warmup-epochs', type=float, default=5,
                    help='number of warmup epochs')

args = parser.parse_args()

# Set `verbose` to `1` if this is the root worker. Otherwise, it should be zero.
if hvd.rank() == 0:
    verbose = 1
else:
    verbose = 0

artists = pd.read_csv('./artists.csv')


def sort_artist_by_of_paintings(artists):
    artists = artists.sort_values(by=['paintings'], ascending=False)
    # Create a dataframe with artists having more than 200 paintings
    artists_top = artists[artists['paintings'] >= 200].reset_index()
    artists_top = artists_top[['name', 'paintings']]
    artists_top['class_weight'] = artists_top.paintings.sum() / (artists_top.shape[0] * artists_top.paintings)
    return artists_top


artists_top = sort_artist_by_of_paintings(artists)
class_weights = artists_top['class_weight'].to_dict()
images_dir = './images/images/'

updated_name = "Albrecht_Du╠êrer"
artists_top.iloc[4, 0] = updated_name
artists_top_name = artists_top['name'].str.replace(' ', '_').values

batch_size = 64 #args.batch_size
train_input_shape = (224, 224, 3)
n_classes = artists_top.shape[0]

train_datagen = ImageDataGenerator(validation_split=0.2,
                                   rescale=1. / 255.,
                                   # rotation_range=45,
                                   # width_shift_range=0.5,
                                   # height_shift_range=0.5,
                                   shear_range=5,
                                   # zoom_range=0.7,
                                   horizontal_flip=True,
                                   vertical_flip=True, )

train_generator = train_datagen.flow_from_directory(directory=images_dir,
                                                    class_mode='categorical',
                                                    target_size=train_input_shape[0:2],
                                                    batch_size=batch_size,
                                                    subset="training",
                                                    shuffle=True,
                                                    classes=artists_top_name.tolist())

valid_generator = train_datagen.flow_from_directory(directory=images_dir,
                                                    class_mode='categorical',
                                                    target_size=train_input_shape[0:2],
                                                    batch_size=batch_size,
                                                    subset="validation",
                                                    shuffle=True,
                                                    classes=artists_top_name.tolist())

STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
STEP_SIZE_VALID = valid_generator.n // valid_generator.batch_size
print("Total number of batches =", STEP_SIZE_TRAIN, "and", STEP_SIZE_VALID)


base_model = ResNet50(weights='imagenet', include_top=False, input_shape=train_input_shape)

for layer in base_model.layers:
    layer.trainable = True
base_model.summary()
classifier = Flatten()(base_model.output)

# Initialize the CNN
classifier = Dense(512, activation='relu')(classifier)
classifier = BatchNormalization()(classifier)

classifier = Dense(16, activation='relu')(classifier)
classifier = BatchNormalization()(classifier)

output = Dense(n_classes, activation='softmax')(classifier)
model = Model(inputs=base_model.input, outputs=output)

early_stop = EarlyStopping(monitor='val_loss', patience=20, verbose=2,
                           mode='auto', restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5,
                              verbose=2, mode='auto')

callbacks = [
    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),
    hvd.callbacks.MetricAverageCallback(),
    early_stop,
    reduce_lr
]


class PrintTotalTime(keras.callbacks.Callback):
    def on_train_begin(self, logs=None):
        self.start_time = time()

    def on_epoch_end(self, epoch, logs=None):
        total_time = round(time() - self.start_time, 2)
        print("Cumulative training time after epoch {}: {}".format(epoch + 1, total_time))

    def on_train_end(self, logs=None):
        total_time = round(time() - self.start_time, 2)
        print("Cumulative training time: {}".format(total_time))


if verbose:
    callbacks.append(PrintTotalTime())

# Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
if hvd.rank() == 0:
    callbacks.append(keras.callbacks.ModelCheckpoint('./checkpoint-{epoch}.h5'))

# Freeze core ResNet layers and train again
for layer in model.layers:
    layer.trainable = False

# У меня вопрос насчет индекса
for layer in model.layers[:50]:
    layer.trainable = True

opt = tf.optimizers.Adam(0.01 * hvd.size())
opt = hvd.DistributedOptimizer(opt)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

n_epoch = 1 # args.epochs

history = model.fit(train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN // hvd.size(),
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID // hvd.size(),
                    epochs= n_epoch,
                    use_multiprocessing=True,
                    workers=hvd.size(),
                    shuffle=True,
                    verbose=verbose,
                    callbacks=callbacks,
                    class_weight=class_weights)


############################################
############################################
############################################


# Freeze core ResNet layers and train again
for layer in model.layers:
    layer.trainable = False

for layer in model.layers[:50]:
    layer.trainable = True

# Compile the CNN
opt = tf.optimizers.Adam(0.0001 * hvd.size())
opt = hvd.DistributedOptimizer(opt)

model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

n_epoch = 1
history2 = model.fit(train_generator,
                     steps_per_epoch=STEP_SIZE_TRAIN // hvd.size(),
                     validation_data=valid_generator,
                     validation_steps=STEP_SIZE_VALID // hvd.size(),
                     epochs=n_epoch,
                     use_multiprocessing=True,
                     shuffle=True,
                     verbose=verbose,
                     callbacks=callbacks,
                     workers=hvd.size(),
                     class_weight=class_weights)

plt.figure(figsize=(10, 10))

plt.subplot(2, 2, 1)
plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Training - Loss Function')
plt.show()

plt.subplot(2, 2, 2)
plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Train - Accuracy')
plt.show()

sys.stderr.close()

