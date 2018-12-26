from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D, MaxPooling2D

def model2D_1(self,inputshape,nw):

    # conv -> relu -> conv -> relu -> pool -> drop -> conv -> relu -> conv
    # -> relu -> pool -> drop -> flat -> dense -> relu -> drop -> dense -> soft

    self.add(Conv2D(nw[0][0], kernel_size=nw[1][0], padding=nw[2][0],
                    input_shape=(inputshape[0],inputshape[1], inputshape[2])))
    self.add(Activation(nw[4][0]))
    self.add(Conv2D(nw[0][1], kernel_size=nw[1][1], padding=nw[2][1]))
    self.add(Activation(nw[4][1]))
    self.add(MaxPooling2D(pool_size=(nw[3][0], nw[3][0])))
    self.add(Dropout(nw[5][0]))
    self.add(Conv2D(nw[0][2], kernel_size=nw[1][2], padding=nw[2][2]))
    self.add(Activation(nw[4][2]))
    self.add(Conv2D(nw[0][3], nw[1][3], nw[2][3]))
    self.add(Activation(nw[4][3]))
    self.add(MaxPooling2D(pool_size=(nw[3][1], nw[3][1])))
    self.add(Dropout(nw[5][1]))
    self.add(Flatten())
    self.add(Dense(nw[6][0]))
    self.add(Activation(nw[4][4]))
    self.add(Dropout(nw[5][2]))
    self.add(Dense(nw[6][1]))
    self.add(Activation(nw[4][5]))
    return self


def model2D_2(self,inputshape,nw):

    # conv -> pool -> relu -> drop -> pool -> flat -> dense -> relu -> drop -> dense -> soft

    self.add(Conv2D(nw[0][0], kernel_size=nw[1][0], padding=nw[2][0],
                    input_shape=(inputshape[0],inputshape[1], inputshape[2])))
    self.add(MaxPooling2D(pool_size=(nw[3][0], nw[3][0])))
    self.add(Activation(nw[4][0]))
    self.add(Dropout(nw[5][0]))
    self.add(MaxPooling2D(pool_size=(nw[3][1], nw[3][1])))
    self.add(Flatten())
    self.add(Dense(nw[6][0]))
    self.add(Activation(nw[4][4]))
    self.add(Dropout(nw[5][2]))
    self.add(Dense(nw[6][1]))
    self.add(Activation(nw[4][5]))
    return self

def model2D_3(self,inputshape,nw):

    # LE-NET
    # conv -> relu -> pool -> conv -> relu -> pool -> flat -> dense -> relu -> dense -> soft

    self.add(Conv2D(nw[0][0], kernel_size=nw[1][0], padding=nw[2][0],
                     input_shape=(inputshape[0],inputshape[1],inputshape[2])))
    self.add(Activation(nw[4][0]))
    self.add(MaxPooling2D(pool_size=(nw[3][0], nw[3][0]), strides=(nw[7][0], nw[7][0])))
    self.add(Conv2D(nw[0][1], kernel_size=nw[1][1], padding=nw[2][1]))
    self.add(Activation(nw[4][1]))
    self.add(MaxPooling2D(pool_size=(nw[3][1], nw[3][1]), strides=(nw[7][1], nw[7][1])))
    self.add(Flatten())
    self.add(Dense(nw[6][0]))
    self.add(Activation(nw[4][2]))
    self.add(Dense(nw[6][1]))
    self.add(Activation(nw[4][3]))
    return self

def model3D_1(self):
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

def model3D_2(self):
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

def model3D_3(self):
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
