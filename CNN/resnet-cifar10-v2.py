import math
import tensorflow as tf
from tensorflow.keras.layers import Dense, Conv2D, BatchNormalization, Activation
from tensorflow.keras.layers import AveragePooling2D, Input, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K
from tensorflow.keras.models import Model
from tensorflow.keras.datasets import cifar10
import numpy as np
import os
import matplotlib.pyplot as plt


(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Image Generator

# ROWS = 10
# x = x_train.astype('uint8')
# fig, axes1 = plt.subplots(ROWS, ROWS, figsize=(10, 10))
# for j in range(ROWS):
#     for k in range(ROWS):
#         i = np.random.choice(range(len(x)))
#         axes1[j][k].set_axis_off()
#         axes1[j][k].imshow(x[i:i+1][0])

# Training parameters
BATCH_SIZE = 32 # orig paper trained all networks with batch_size = 128
EPOCHS = 100 # 200
USE_AUGMENTATION = True
NUM_CLASSES = np.unique(y_train).shape[0]
COLORS = x_train.shape[3]

# Subtracting pixel mean improves Accuracy
SUBTRACT_PIXEL_MEAN = True

# Model Version
# Orig paper : version = 1 (ResNet v1), Improved ResNet: version = 2 (ResNet v2)
VERSION = 2

if VERSION == 1:
    DEPTH = COLORS * 6 + 2
elif VERSION == 2:
    DEPTH = COLORS * 9 + 2

def lr_schedule(epoch):

    lr = 1e-3
    if epoch > 180:
        lr *= 0.5e-3
    elif epoch > 160:
        lr *= 1e-3
    elif epoch > 120:
        lr *= 1e-2
    elif epoch > 80:
        lr *= 1e-1
    print("Learning rate: ", lr)
    return lr

def resnet_layer(inputs, num_filters=16, kernel_size=3, strides=1, activation='relu', batch_normalization=True, conv_first=True):
    conv = Conv2D(num_filters, kernel_size=kernel_size, strides=strides, padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(1e-4))

    x = inputs
    if conv_first:
        x = conv(x)
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
    else:
        if batch_normalization:
            x = BatchNormalization()(x)
        if activation is not None:
            x = Activation(activation)(x)
        x = conv(x)
    return x

def resnet_v1(input_shape, depth, num_classes=10):
    if (depth - 2) % 6 != 0:
        raise ValueError ('Depth should be 6n+2 (eg 20, 32, 44 in [a])')

    # Start model definition
    num_filters = 16
    num_res_blocks = int((depth - 2) / 6)
    # depth : 20 -> num_res_blocks : 3

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs)

    # Instantiate the stack of residual units
    for stack in range(3):
        for res_block in range(num_res_blocks):
            strides = 1
            if stack > 0 and res_block == 0: # first layer but not first stack
                strides = 2 #downsample
            y = resnet_layer(inputs=x, num_filters=num_filters, strides=strides)
            y = resnet_layer(inputs=y, num_filters=num_filters, activation=None)

            if stack > 0 and res_block == 0:
                # first layer but not first stack
                # linear projection residual shortcut connection to match
                # chaned dims
                x = resnet_layer(inputs=x, num_filters=num_filters, kernel_size=1, strides=strides, activation=None, batch_normalization=False)
            x = tensorflow.keras.layers.add([x, y])
            x = Activation('relu')(x)
        num_filters *= 2

    # Add classifier on top.
    # v1 does not use BN after last shortcut connection - ReLU
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes, activation='softmax', kernel_initializer='he_normal')(y)

    # Instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model

# The primary difference of the full preactivation 'v2' variant compared to the
# 'v1' variant is the use of batch normalization before every weight layer
def resnet_v2(input_shape, depth, num_classes=10):
    if (depth-2) % 9 != 0:
        raise ValueError('depth should be 9n+2 (eg 110 in [b])')

    # start model definition
    num_filters_in = 16
    num_res_blocks = int((depth - 2) / 9)

    inputs = Input(shape=input_shape)
    x = resnet_layer(inputs=inputs, num_filters=num_filters_in, conv_first=True)

    # instantiate the stack of residual units
    for stage in range(3):
        for res_block in range(num_res_blocks):
            activation = 'relu'
            batch_normalization = True
            strides = 1
            if stage == 0:
                num_filters_out = num_filters_in * 4
                # first layer and first stage
                if res_block == 0:
                    activation = None
                    batch_normalization = None
            else:
                num_filters_out = num_filters_in * 2
                # first layer but not first stage
                if res_block == 0:
                    # downsample
                    strides = 2

            # bottleneck residual unit
            y = resnet_layer(inputs=x, num_filters=num_filters_in, kernel_size=1, strides=strides, activation=activation, batch_normalization=batch_normalization, conv_first=False)
            y = resnet_layer(inputs=y, num_filters=num_filters_in, conv_first=False)
            y = resnet_layer(inputs=y, num_filters=num_filters_out, kernel_size=1, conv_first=False)

            if res_block == 0:
                # linear projection residual shortcut connection
                # to match chaged dims
                x = resnet_layer(inputs=x, num_filters=num_filters_out, kernel_size=1, strides=strides, activation=None, batch_normalization=False)
            x = tf.keras.layers.add([x, y])

        num_filters_in = num_filters_out

    # add classifier on top.
    # v2 has BN-ReLU before Pooling
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = AveragePooling2D(pool_size=8)(x)
    y = Flatten()(x)
    outputs = Dense(num_classes,
                    activation='softmax',
                    kernel_initializer='he_normal')(y)

    # instantiate model.
    model = Model(inputs=inputs, outputs=outputs)
    return model


# Input image dimensions.
input_shape = x_train.shape[1:]

# Normalize data
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

# If subtract pixel mean is enabled
if SUBTRACT_PIXEL_MEAN:
    x_train_mean = np.mean(x_train, axis=0)
    x_train -= x_train_mean
    x_test -= x_train_mean

print('x_train shape: ', x_train.shape)
print('y_train shape: ', y_train.shape)
print('train samples ', x_train.shape[0])
print('test samples ', x_test.shape[0])

y_train = tf.keras.utils.to_categorical(y_train, NUM_CLASSES)
y_test = tf.keras.utils.to_categorical(y_test, NUM_CLASSES)

# Create the neural network
if VERSION == 2:
    model = resnet_v2(input_shape=input_shape, depth=DEPTH)
elif VERSION == 1:
    model = resnet_v1(input_shape=input_shape, depth=DEPTH)

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=lr_schedule(0)), metrics=['accuracy'])


import time

start_time = time.time()

# Prepare callbacks for model saving and for learning rate adjustment
lr_scheduler = LearningRateScheduler(lr_schedule)
# ReduceLROnPlateau : learning rate increase or decrease
# monitor='val_loss' : val acc callback
# factor=0.5 : above not inproved val_loss --> callback learning rate *= 0.5
# patience=10 : during epoch 10 not improve val_loss --> callback
# cooldown : after reduced lr, Number of epochs to wait before resuming normal operation

lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
callbacks = [lr_reducer, lr_scheduler]

# Run training, with or without data augmentation
if not USE_AUGMENTATION:
    print("Not Using data augmentation")
    model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=EPOCHS, validation_data=(x_test, y_test), shffle=True, callbacks=call_backs)
else:
    print("Using data augmentation")
    datagen = ImageDataGenerator(
                                 # set input mean to 0 over the dataset
                                 featurewise_center=False,
                                 # set each sample mean to 0
                                 samplewise_center=False,
                                 # divide inputs by std of dataset
                                 featurewise_std_normalization=False,
                                 # divide each input by its str
                                 samplewise_std_normalization=False,
                                 # apply ZCA whitening
                                 zca_whitening=False,
                                 # epsilon for ZCA whitening
                                 zca_epsilon=1e-06,
                                 # randomly rotate images in the range (deg 0 to 180))
                                 rotation_range=0,
                                 # randomly shift image horizontally
                                 width_shift_range=0.1,
                                 # randomly shift image vertically
                                 height_shift_range=0.1,
                                 # set range for random shear
                                 shear_range=0.,
                                 # set range for random zoom
                                 zoom_range=0.,
                                 # set range for channel shifts
                                 channel_shift_range=0.,
                                 # set mode for filling points outside the input boundaries
                                 fill_mode='nearest',
                                 # value used for fill_mode = 'constant'
                                 cval=0.,
                                 # randomly flip images
                                 horizontal_flip=True,
                                 vertical_flip=False,
                                 # set rescaling factor (applied before any other transformation)
                                 rescale=None,
                                 # image data format, either "channel_first" or "channels_last"
                                 data_format=None,
                                 # faction of images reserved for validation (strictly between 0 and 1)
                                 validation_split=0.0)
    # compute quantities required for featurewise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    steps_per_epoch =  math.ceil(len(x_train) / BATCH_SIZE)
    # fit the model on the batches generated by datagen.flow().
    model.fit(x=datagen.flow(x_train, y_train, batch_size=BATCH_SIZE),
              verbose=1,
              epochs=EPOCHS,
              validation_data=(x_test, y_test),
              steps_per_epoch=steps_per_epoch,
              callbacks=callbacks)


# score trained model
scores = model.evaluate(x_test,
                        y_test,
                        batch_size=BATCH_SIZE,
                        verbose=0)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])

elapsed_time = time.time() - start_time
print("Elapsed time: {}".format((elapsed_time)))
