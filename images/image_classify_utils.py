import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SHAPE = 150
NUM_CLASS = 3

def gen_image(train_dir, val_dir, batch_size=100, val_split=0.2, transform_data=False):
    if transform_data:
        image_gen_train = ImageDataGenerator(
            rescale=1./255,
            rotation_range=45,
            width_shift_range=.15,
            height_shift_range=.15,
            horizontal_flip=True,
            zoom_range=0.5,
            validation_split=val_split
        )
    else:
        image_gen_train = ImageDataGenerator(rescale=1./255,
                                             validation_split=val_split)

    train_data_gen = image_gen_train.flow_from_directory(
        batch_size=batch_size,
        directory=train_dir,
        shuffle=True,
        target_size=(IMG_SHAPE,IMG_SHAPE),
        subset='training',
        class_mode='categorical'
    )
    image_gen_val = ImageDataGenerator(rescale=1./255,
                                       validation_split=val_split)

    val_data_gen = image_gen_val.flow_from_directory(
        batch_size=batch_size,
        directory=val_dir,
        shuffle=True,
        target_size=(IMG_SHAPE, IMG_SHAPE),
        subset='validation',
        class_mode='categorical'
    )
    return (train_data_gen, val_data_gen)


def build_model(train_data_gen, val_data_gen, epochs, dropout=False):
    model = Sequential()
    model.add(Conv2D(16, 3, padding='same', activation='relu', input_shape=(150,150,3,)))
    model.add(MaxPooling2D(pool_size=2))
    if dropout:
        model.add(Dropout(0.3))
    model.add(Conv2D(32, 3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))

    model.add(Conv2D(64, 3, padding='same', activation='relu'))
    model.add(MaxPooling2D(pool_size=2))

    if dropout:
        model.add(Dropout(0.3))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    if dropout:
        model.add(Dropout(0.1))
    model.add(Dense(3, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy']
    )
    history = model.fit_generator(
        train_data_gen,
        epochs=epochs,
        validation_data=val_data_gen,
        verbose=0
#         steps_per_epoch=1,
#         validation_steps=1
    )
    return (model, history)
    
def viz_results(history, epochs):
    
    acc = history.history['acc']
    val_acc = history.history['val_acc']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(epochs)

    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()
    return (acc, val_acc)
