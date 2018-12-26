from keras.models import Sequential
from keras.models import model_from_json
from keras.optimizers import SGD

def init():
    init = Sequential()
    return init

def structure(model):
    a = model.layers
    b = model.outputs
    c = model.inputs
    d = model.get_config()
    return a,b,c,d

def train(dataset,parameters, model):
    model.compile(loss='categorical_crossentropy', optimizer=parameters[4],
                  metrics=['accuracy'])
    model.summary()
    model.fit(dataset[0], dataset[1], batch_size=parameters[0],
                    epochs=parameters[1], validation_split=parameters[2],
                    verbose=parameters[3])
    score = model.evaluate(dataset[2],dataset[3],batch_size=parameters[0],verbose=parameters[3])
    return score

def savemodel(model,jfile,hfile):
    json = model.to_json()
    open(jfile,'w').write(json)
    model.save_weights(hfile,overwrite= True)

def maintrain(inputhsape,parameters,dataset,nw_parameters,func,jfile,hfile):
    initial = init()
    layer = func(initial,inputhsape,nw_parameters)
    train(dataset,parameters,layer)
    savemodel(initial, jfile, hfile)
    yapi = structure(layer)
    return initial,layer,yapi

def runmodel(arch,weights,imgs):
    model_architecture = arch
    model_weights = weights
    model = model_from_json(open(model_architecture).read())
    model.load_weights(model_weights)
    optim = SGD()
    model.compile(loss='categorical_crossentropy', optimizer=optim,
              metrics=['accuracy'])
    predictions = model.predict_classes(imgs)
    print(predictions)