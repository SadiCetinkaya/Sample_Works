from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop

from keras.preprocessing.image import ImageDataGenerator

# from quiver_engine import server

# import matplotlib.pyplot as plt

# CIFAR_10 is a set of 60K images 32x32 pixels on 3 channels
IMG_CHANNELS = 3
IMG_ROWS = 32
IMG_COLS = 32

# constant
BATCH_SIZE = 128
NB_EPOCH = 40
NB_CLASSES = 10
VERBOSE = 1
VALIDATION_SPLIT = 0.2
OPTIM = RMSprop()

# load dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert to categorical
Y_train = np_utils.to_categorical(y_train, NB_CLASSES)
Y_test = np_utils.to_categorical(y_test, NB_CLASSES)
print(X_test[1])

# float and normalization
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(X_test[1])

# network parameters
n_filter = [32, 32, 64, 64]
n_conv = [3, 3, 3, 3]
n_padding = ['valid', 'same', 'same',  3]
n_pool = [2, 2]
n_activation = ['relu', 'relu', 'relu', 'relu', 'relu', 'softmax']
n_dropout = [0.25, 0.25, 0.5]
n_dense = [512, NB_CLASSES]

# networks
def model1():
    model = Sequential()
    model.add(Conv2D(n_filter[0], kernel_size=n_conv[0], padding=n_padding[0],
                 input_shape=(IMG_ROWS, IMG_COLS, IMG_CHANNELS)))
    model.add(Activation(n_activation[0]))
    model.add(Conv2D(n_filter[1], kernel_size=n_conv[1], padding=n_padding[1]))
    model.add(Activation(n_activation[1]))
    model.add(MaxPooling2D(pool_size=(n_pool[0], n_pool[0])))
    model.add(Dropout(n_dropout[0]))
    model.add(Conv2D(n_filter[2], kernel_size=n_conv[2], padding=n_padding[2]))
    model.add(Activation(n_activation[2]))
    model.add(Conv2D(n_filter[3], n_conv[3], n_padding[3]))
    model.add(Activation(n_activation[3]))
    model.add(MaxPooling2D(pool_size=(n_pool[1], n_pool[1])))
    model.add(Dropout(n_dropout[1]))
    model.add(Flatten())
    model.add(Dense(n_dense[0]))
    model.add(Activation(n_activation[4]))
    model.add(Dropout(n_dropout[2]))
    model.add(Dense(n_dense[1]))
    model.add(Activation(n_activation[5]))
    return model

def model2():
    model = Sequential()
    model.add(Conv2D(n_filter[0], kernel_size=n_conv[0], padding=n_padding[0],
                 input_shape=(IMG_ROWS, IMG_COLS, IMG_CHANNELS)))
    model.add(Activation(n_activation[0]))
    model.add(MaxPooling2D(pool_size=(n_pool[0], n_pool[0])))
    model.add(Dropout(n_dropout[0]))
    model.add(Flatten())
    model.add(Dense(n_dense[0]))
    model.add(Activation(n_activation[4]))
    model.add(Dropout(n_dropout[2]))
    model.add(Dense(n_dense[1]))
    model.add(Activation(n_activation[5]))
    return model

def summary(self):
    self.compile(loss='categorical_crossentropy', optimizer=OPTIM,
                  metrics=['accuracy'])
    self.summary()

model = model1()
summary = summary(model)
#model.summary()

datagen = ImageDataGenerator(
    featurewise_center=False,  # set input mean to 0 over the dataset
    samplewise_center=False,  # set each sample mean to 0
    featurewise_std_normalization=False,  # divide inputs by std of the dataset
    samplewise_std_normalization=False,  # divide each input by its std
    zca_whitening=False,  # apply ZCA whitening
    rotation_range=40,  # randomly rotate images in the range (degrees, 0 to 180)
    width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
    height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
    horizontal_flip=True,  # randomly flip images
    fill_mode='nearest',
    vertical_flip=True)  # randomly flip images

datagen.fit(X_train)

# train

history = model.fit(X_train, Y_train, batch_size=BATCH_SIZE,
                    epochs=NB_EPOCH, validation_split=VALIDATION_SPLIT,
                    verbose=VERBOSE)

# model.fit_generator(datagen.flow(X_train, Y_train,
#                        batch_size=BATCH_SIZE),
#                        samples_per_epoch=X_train.shape[0],
#                        nb_epoch=NB_EPOCH,
#                        verbose=VERBOSE)

# server.launch(model)

print('Testing...')
score = model.evaluate(X_test, Y_test,
                       batch_size=BATCH_SIZE, verbose=VERBOSE)
print("\nTest score:", score[0])
print('Test accuracy:', score[1])
print(score)

# save model
model_json = model.to_json()
open('cifar10_v1_architecture.json', 'w').write(model_json)
model.save_weights('cifar10_v1_weights.h5', overwrite=True)

l1 = model.layers[0]
w1 = l1.get_weights()
print(w1)

# list all data in history
#print(history.history.keys())
# summarize history for accuracy
# plt.plot(history.history['acc'])
# plt.plot(history.history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
## summarize history for loss
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
