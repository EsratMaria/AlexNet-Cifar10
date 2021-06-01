import os
import keras
from keras.datasets import cifar10
from keras import backend as K
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dense, BatchNormalization, Activation, MaxPooling2D
from keras.models import Model
from keras.layers import concatenate, Dropout, Flatten

from keras import optimizers, regularizers
from keras.preprocessing.image import ImageDataGenerator
from keras.initializers import he_normal
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
import matplotlib.pyplot as plt

num_classes = 10
batch_size = 64
epochs = 25
iterations = 782
DROPOUT = 0.5  # keeping 50%
CONCAT_AXIS = 3
DATA_FORMAT = 'channels_last'
log_filepath = 'C:\\Users\\Esrat Maria\\Desktop\\AlexNet'


print('----------------------------------------------------------------------------------------------------------')
print('Deep Learning Framework : Keras', keras.__version__)
print('Number of Classes : ', +num_classes)
print('Data Augmentation : Yes, Image Data Generator.')
print('Learning Rate : 0.01 (Depending upon the number of epoch. )')
print('Batch Number : ', +batch_size)
print('Momentum : 0.9')
print('Dropout Rate : ', +DROPOUT)
print('Epoch number should not be less than 15. After 13 epochs the system hits accuracy of 0.7(70%)')
print('With 25 epochs the accuracy is above 80%')
print('----------------------------------------------------------------------------------------------------------')


def color_preprocessing(x_train, x_test):
  x_train = x_train.astype('float32')
  x_test = x_test.astype('float32')
  mean = [125.307, 122.95, 113.865]
  std = [62.9932, 62.0887, 66.7048]
  for i in range(3):
    x_train[:, :, :, i] = (x_train[:, :, :, i] - mean[i]) / std[i]
    x_test[:, :, :, i] = (x_test[:, :, :, i] - mean[i]) / std[i]
  return x_train, x_test

# defining learning rate based on the number of epoch


def scheduler(epoch):
  if epoch < 100:
    return 0.01
  if epoch < 200:
    return 0.001
  return 0.0001


# loading cifar10 data                                                                               
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)
x_train, x_test = color_preprocessing(x_train, x_test)

print(x_train.shape)
# Building CNN (AlexNet)                                                                             


def alexnet(img_input, classes=10):
  # 1st conv layer
  x = Conv2D(96, (11, 11), strides=(4, 4), padding='same',
             activation='relu', kernel_initializer='uniform')(img_input)  # valid                                   
  x = MaxPooling2D(pool_size=(3, 3), strides=(
      2, 2), padding='same', data_format=DATA_FORMAT)(x)
  x = BatchNormalization()(x)

  # 2nd conv layer
  x = Conv2D(256, (5, 5), strides=(1, 1), padding='same',
             activation='relu', kernel_initializer='uniform')(x)
  x = MaxPooling2D(pool_size=(3, 3), strides=(
      2, 2), padding='same', data_format=DATA_FORMAT)(x)
  x = BatchNormalization()(x)

  # 3rd conv layer
  x = Conv2D(384, (3, 3), strides=(1, 1), padding='same',
             activation='relu', kernel_initializer='uniform')(x)
  x = BatchNormalization()(x)

  # 4th conv layer
  x = Conv2D(384, (3, 3), strides=(1, 1), padding='same',
             activation='relu', kernel_initializer='uniform')(x)
  x = BatchNormalization()(x)

  # 5th conv layer
  x = Conv2D(256, (3, 3), strides=(1, 1), padding='same',
             activation='relu', kernel_initializer='uniform')(x)
  x = MaxPooling2D(pool_size=(3, 3), strides=(
      2, 2), padding='same', data_format=DATA_FORMAT)(x)
  x = BatchNormalization()(x)

  # flattening before sending to fully connected layers
  x = Flatten()(x)
  # fully connected layers
  x = Dense(4096, activation='relu')(x)
  x = Dropout(0.5)(x)
  x = BatchNormalization()(x)
  x = Dense(1000, activation='relu')(x)
  x = Dropout(0.5)(x)
  x = BatchNormalization()(x)

  # output layer
  out = Dense(classes, activation='softmax')(x)
  return out


# defining input image size according to cifar10
img_input = Input(shape=(32, 32, 3))
output = alexnet(img_input)
model = Model(img_input, output)
model.summary()


# set optimizer
sgd = optimizers.SGD(lr=.001, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy',
              optimizer=sgd, metrics=['accuracy'])
# set callback
tb_cb = TensorBoard(log_dir=log_filepath, histogram_freq=0)
change_lr = LearningRateScheduler(scheduler)
cbks = [change_lr, tb_cb]

# set data augmentation
datagen = ImageDataGenerator(horizontal_flip=True,
                             width_shift_range=0.125,
                             height_shift_range=0.125,
                             fill_mode='constant', cval=0.)
datagen.fit(x_train)


# start training

history = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                              steps_per_epoch=iterations,
                              epochs=epochs,
                              callbacks=cbks,
                              validation_data=(x_test, y_test))

# plotting loss VS epoch
plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.title('model loss')
plt.ylabel('training and test loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()


# plotting accuracy VS epoch
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.title('model accuracy')
plt.xlabel('Epoch')
plt.ylabel('Training and Test Accuracy')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.show()

print('loading the final test accuracy.....')

test_loss, test_acc = model.evaluate(x_test,  y_test, verbose=2)
train_loss, train_acc = model.evaluate(x_train,  y_train, verbose=2)
# final accuracies
print('Training Accuracy', +train_acc)
print('Test Accuracy', +test_acc)
