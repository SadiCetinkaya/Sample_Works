from keras.datasets import cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import ImageDataGenerator

#input shape
IMG_CHANNELS = 3
IMG_ROWS = 32
IMG_COLS = 32

#model parameters
BATCH_SIZE = 128
NB_EPOCH = 2
NB_CLASSES = 10
VERBOSE = 1
VALIDATION_SPLIT = 0.2
OPTIM = RMSprop()

#dataset
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

inputshape = [IMG_ROWS,IMG_COLS,IMG_CHANNELS]
dataset = [X_train,Y_train,X_test,Y_test]
parameters = [BATCH_SIZE, NB_EPOCH,VALIDATION_SPLIT,VERBOSE]

def init():
    init = Sequential()
    return init

def model1(self):
    self.add(Conv2D(n_filter[0], kernel_size=n_conv[0], padding=n_padding[0],
            input_shape=(IMG_ROWS, IMG_COLS, IMG_CHANNELS)))
    self.add(Activation(n_activation[0]))
    self.add(Conv2D(n_filter[1], kernel_size=n_conv[1], padding=n_padding[1]))
    self.add(Activation(n_activation[1]))
    self.add(MaxPooling2D(pool_size=(n_pool[0], n_pool[0])))
    self.add(Dropout(n_dropout[0]))
    self.add(Conv2D(n_filter[2], kernel_size=n_conv[2], padding=n_padding[2]))
    self.add(Activation(n_activation[2]))
    self.add(Conv2D(n_filter[3], n_conv[3], n_padding[3]))
    self.add(Activation(n_activation[3]))
    self.add(MaxPooling2D(pool_size=(n_pool[1], n_pool[1])))
    self.add(Dropout(n_dropout[1]))
    self.add(Flatten())
    self.add(Dense(n_dense[0]))
    self.add(Activation(n_activation[4]))
    self.add(Dropout(n_dropout[2]))
    self.add(Dense(n_dense[1]))
    self.add(Activation(n_activation[5]))
    return self

def model2(self):
    self.add(Conv2D(n_filter[0], kernel_size=n_conv[0], padding=n_padding[0],
            input_shape=(IMG_ROWS, IMG_COLS, IMG_CHANNELS)))
    self.add(MaxPooling2D(pool_size=(n_pool[0], n_pool[0])))
    self.add(Activation(n_activation[0]))
    self.add(Dropout(n_dropout[0]))
    self.add(MaxPooling2D(pool_size=(n_pool[0], n_pool[0])))
    self.add(Flatten())
    self.add(Dense(n_dense[0]))
    self.add(Activation(n_activation[4]))
    self.add(Dropout(n_dropout[2]))
    self.add(Dense(n_dense[1]))
    self.add(Activation(n_activation[5]))
    return self

def summary(model):
    model.compile(loss='categorical_crossentropy', optimizer=OPTIM,
                  metrics=['accuracy'])
    model.summary()

def structure(model):
    a = model.layers
    b = model.outputs
    c = model.inputs
    d = model.get_config()
    return a,b,c,d

def train(dataset,parameters, model):
    model.fit(dataset[0], dataset[1], batch_size=parameters[0],
                    epochs=parameters[1], validation_split=parameters[2],
                    verbose=parameters[3])
    score = model.evaluate(dataset[2],dataset[3],batch_size=parameters[0],verbose=parameters[3])
    return score

def savemodel(model,jfile,hfile):
    json = model.to_json()
    open(jfile,'w').write(json)
    model.save_weights(hfile,overwrite= True)

def mainrun(func):
    initial = init()
    layer = func(initial)
    ozet = summary(layer)
    train(dataset,parameters,layer)
    yapi = structure(layer)
    return initial,layer,ozet,yapi

if __name__ == '__main__':
    results = mainrun(model1)
#    mainrun(model2)
#    print(results[2])
    print(results)

#w1 = layer_1.layers[1]
#w2 = w1.get_weights()
#print(w2)