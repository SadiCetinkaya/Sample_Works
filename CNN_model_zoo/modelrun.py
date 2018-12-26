from keras.datasets import cifar10
from keras.utils import np_utils
from keras.optimizers import SGD, Adam, RMSprop
from keras.preprocessing.image import ImageDataGenerator
from modelclass import *

# input shape
IMG_CHANNELS = 3
IMG_1d = 32
IMG_2d = 32

# model parameters
BATCH_SIZE = 128
NB_EPOCH = 2
NB_CLASSES = 10
VERBOSE = 1
VALIDATION_SPLIT = 0.2
OPTIM = RMSprop()

# dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()
print('X_train shape:', X_train.shape)
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')
Y_train = np_utils.to_categorical(y_train, NB_CLASSES)
Y_test = np_utils.to_categorical(y_test, NB_CLASSES)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

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

# network parameters
n_filter = [32, 32, 64, 64]
n_conv = [4, 3, 3, 3]
n_padding = ['valid', 'same', 'same',  3]
n_pool = [2, 2]
n_activation = ['relu', 'relu', 'relu', 'relu', 'relu', 'softmax']
n_dropout = [0.25, 0.25, 0.5]
n_dense = [512, NB_CLASSES]
n_strides = [2,2]

# all metrics
inputshape = [IMG_1d,IMG_2d,IMG_CHANNELS]
parameters = [BATCH_SIZE, NB_EPOCH,VALIDATION_SPLIT,VERBOSE,OPTIM]
dataset = [X_train,Y_train,X_test,Y_test]
nw_parameters = [n_filter,n_conv,n_padding, n_pool, n_activation, n_dropout, n_dense,n_strides]

jfile = 'cifar10_v1_architecture.json'
hfile = 'cifar10_v1_weights.h5'

if __name__ == '__main__':
    results = maintrain(inputshape,parameters,dataset,nw_parameters,model2D_1,jfile,hfile)
#    mainrun(model2)
#    print(results[2])
    print(results)
