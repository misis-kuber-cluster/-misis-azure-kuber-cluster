import pandas as pd
from keras.applications.resnet import ResNet50
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os
from keras.layers import Flatten, Dense, BatchNormalization
from keras.models import Model
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.preprocessing.image import ImageDataGenerator
import DatasetValidation


# DatasetValidation.download_dataset() - call if dataset isn't downloaded.

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

artists = pd.read_csv('./data/artists.csv')
print(artists.shape)


def sort_artist_by_of_paintings(artists):
    artists = artists.sort_values(by=['paintings'], ascending=False)
    # Create a dataframe with artists having more than 200 paintings
    artists_top = artists[artists['paintings'] >= 200].reset_index()
    artists_top = artists_top[['name', 'paintings']]
    artists_top['class_weight'] = artists_top.paintings.sum() / (artists_top.shape[0] * artists_top.paintings)
    return artists_top


artists_top = sort_artist_by_of_paintings(artists)

class_weights = artists_top['class_weight'].to_dict()

'''
В датасете ошибка в именовании папки с изображениями у этого автора
'''
updated_name = "Albrecht_Du╠êrer"
artists_top.iloc[4, 0] = updated_name

images_dir = './data/images/images/'
artists_top_name = artists_top['name'].str.replace(' ', '_').values

# Validation
"""
DatasetValidation.is_all_directories_exist(artists_top, images_dir)
"""
DatasetValidation.PlotImages("Vincent van Gogh", "data/images/images/Vincent_van_Gogh/**")


batch_size = 64
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
                                   vertical_flip=True,
                                   )

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

# Compile the CNN
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

early_stop = EarlyStopping(monitor='val_loss', patience=20, verbose=1,
                           mode='auto', restore_best_weights=True)

reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5,
                              verbose=1, mode='auto')

n_epoch = 10
history = model.fit(train_generator, steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=valid_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=n_epoch,
                    shuffle=True,
                    verbose=1,
                    use_multiprocessing=True,
                    callbacks=[reduce_lr],

                    workers=16,
                    class_weight=class_weights)

# Freeze core ResNet layers and train again
for layer in model.layers:
    layer.trainable = False

for layer in model.layers[:50]:
    layer.trainable = True

optimizer = Adam(lr=0.0001)

model.compile(loss='categorical_crossentropy',
              optimizer=optimizer,
              metrics=['accuracy'])

n_epoch = 50
history2 = model.fit(train_generator, steps_per_epoch=STEP_SIZE_TRAIN,
                     validation_data=valid_generator,
                     validation_steps=STEP_SIZE_VALID,
                     epochs=n_epoch,
                     shuffle=True,
                     verbose=1,
                     callbacks=[reduce_lr, early_stop],
                     use_multiprocessing=True,
                     workers=16,
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
plt.legend()
plt.title('Train - Accuracy')

plt.show()

score = model.evaluate_generator(train_generator, verbose=1)
print("Prediction accuracy on train data =", score[1])

score = model.evaluate_generator(valid_generator, verbose=1)
print("Prediction accuracy on test data =", score[1])