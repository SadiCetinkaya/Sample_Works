from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv3D, MaxPooling3D
from keras.optimizers import SGD, Adam, RMSprop

#Input shape
IMG_CHANNELS = 1
IMG_1d = 256
IMG_2d = 256
IMG_3d = 166
OPTIM = RMSprop()

#model parameters
BATCH_SIZE = 128
NB_EPOCH = 40
NB_CLASSES = 3
VERBOSE = 1
VALIDATION_SPLIT = 0.2
OPTIM = RMSprop()

# network parameters
n_filter = [32, 32, 64, 64]
n_conv = [4, 3, 3, 3]
n_padding = ['valid', 'same', 'same',  3]
n_pool = [2, 2]
n_activation = ['relu', 'relu', 'relu', 'relu', 'relu', 'softmax']
n_dropout = [0.25, 0.25, 0.5]
n_dense = [64, NB_CLASSES]

inputshape = [IMG_1d,IMG_2d,IMG_3d,IMG_CHANNELS]
#dataset = [X_train,Y_train,X_test,Y_test]
parameters = [BATCH_SIZE, NB_EPOCH,VALIDATION_SPLIT,VERBOSE]

# Model Functions
def init():
    init = Sequential()
    return init

def model1(self):
    self.add(Conv3D(n_filter[0],kernel_size=n_conv[0], padding=n_padding[1],
            input_shape=(inputshape[0], inputshape[1], inputshape[2], inputshape[3])))
    self.add(Activation(n_activation[0]))
    self.add(MaxPooling3D())
    self.add(Dropout(n_dropout[0]))

    self.add(Flatten())
    self.add(Dense(n_dense[0]))
    self.add(Activation(n_activation[4]))
    self.add(Dropout(n_dropout[2]))
    self.add(Dense(n_dense[1]))
    self.add(Activation(n_activation[5]))
    return self

def model2(self):
    self.add(Conv3D(n_filter[0], kernel_size=n_conv[0], padding=n_padding[0],
                     input_shape=(inputshape[0], inputshape[1], inputshape[2], inputshape[3])))
    self.add(Activation(n_activation[0]))
    self.add(MaxPooling3D())
    self.add(Dropout(n_dropout[0]))

    self.add(Conv3D(n_filter[0], kernel_size=n_conv[1], padding=n_padding[0]))
    self.add(Activation(n_activation[0]))
    self.add(MaxPooling3D())
    self.add(Dropout(n_dropout[0]))

    self.add(Flatten())
    self.add(Dense(n_dense[0]))
    self.add(Activation(n_activation[4]))
    self.add(Dropout(n_dropout[2]))
    self.add(Dense(n_dense[1]))
    self.add(Activation(n_activation[5]))
    return self

def model3(self):
    self.add(Conv3D(n_filter[0], kernel_size=n_conv[1], padding=n_padding[1],
                input_shape=(inputshape[0], inputshape[1], inputshape[2], inputshape[3])))
    self.add(Activation(n_activation[0]))
    self.add(MaxPooling3D())
    self.add(Dropout(n_dropout[0]))

    self.add(Conv3D(n_filter[0], kernel_size=n_conv[1], padding=n_padding[1]))
    self.add(Activation(n_activation[0]))
    self.add(MaxPooling3D())
    self.add(Dropout(n_dropout[0]))

    self.add(Conv3D(n_filter[0], kernel_size=n_conv[1], padding=n_padding[1]))
    self.add(Activation(n_activation[0]))
    self.add(MaxPooling3D())
    self.add(Dropout(n_dropout[0]))

    self.add(Flatten())
    self.add(Dense(n_dense[0]))
    self.add(Activation(n_activation[4]))
    self.add(Dropout(n_dropout[2]))
    self.add(Dense(n_dense[1]))
    self.add(Activation(n_activation[5]))
    return self

def summary(self):
    self.compile(loss='categorical_crossentropy', optimizer=OPTIM,
                  metrics=['accuracy'])
    self.summary()

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
#    train(dataset,parameters,layer)
    yapi = structure(layer)
    return initial,layer,ozet,yapi

if __name__ == '__main__':
     mainrun(model1)
     mainrun(model2)
     mainrun(model3)
#    print(results[2])
#    print(results[3][3])
