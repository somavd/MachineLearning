from __future__ import print_function 
import lasagne
import matplotlib.pyplot as plt
import numpy as np
import theano
import theano.tensor as T
import time
import os
from PIL import Image, ImageFilter
 
def load_dataset(filename):    
    f=open(filename,"r")
    data=f.read()
    data=data.replace('\n',',')
    data=eval(data)
    data=np.reshape(data,(-1,18,18))
    data=data.reshape(-1,1,18,18)
    data=data/np.float32(256)
    #data/np.float32(256)
    op=[]
    n=10
    #n is number of samples of each gesture
    for i in range(n):
        for j in range(10):
            op=np.insert(op,len(op),j)
    a=op
    for i in range(9):
        op=np.insert(op,len(op),a)
    f.close()
    print(op)
    print(type(op[0]))
    return data,op
def load_testdata(filename):    
    f=open(filename,"r")
    data=f.read()
    data=data.replace('\n',',')
    data=eval(data)
    print(len(data))
    data=np.reshape(data,(-1,18,18))
    data=data.reshape(-1,1,18,18)
    data=data/np.float32(256)
    f.close()
    return data

def load_testoutput(filename):    
    f=open(filename,"r")
    data=f.read()
    data=data.replace('\n',',')
    data=list(eval(data))
    return data

def build_cnn(input_var=None):
    l_in = lasagne.layers.InputLayer(shape=(None, 1, 18, 18),
                                        input_var=input_var)
    print(l_in.shape)
    l_c1 = lasagne.layers.Conv2DLayer(
            l_in, num_filters=32, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify,
            W=lasagne.init.GlorotUniform())
    print(l_c1.output_shape)
    l_p1 = lasagne.layers.MaxPool2DLayer(l_c1, pool_size=(2, 2))
    print(l_p1.output_shape)
    l_c2 = lasagne.layers.Conv2DLayer(
            l_p1, num_filters=32, filter_size=(3, 3),
            nonlinearity=lasagne.nonlinearities.rectify)
    print(l_c2.output_shape)
    l_p2 = lasagne.layers.MaxPool2DLayer(l_c2, pool_size=(2, 2))
    print(l_p2.output_shape)
    l_fc = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(l_p2, p=.5),
            num_units=288,
            nonlinearity=lasagne.nonlinearities.rectify)
    print(l_fc.output_shape)
    #print("Hello")
    l_out = lasagne.layers.DenseLayer(
            lasagne.layers.dropout(l_fc, p=.5),
            num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)
    print(l_out.output_shape)
    return l_out

def iterate_minibatches(inputs, targets, batchsize, shuffle=False):
    print(len(inputs))
    print(len(targets))
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)
    for start_idx in range(0, len(inputs) - batchsize + 1, batchsize):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batchsize]
        else:
            excerpt = slice(start_idx, start_idx + batchsize)
        yield inputs[excerpt], targets[excerpt]
        
def imageprepare(argv):
    im = Image.open(argv).convert('L')
    width = float(im.size[0])
    height = float(im.size[1])
    newImage = Image.new('L', (28, 28), (255))
    if width > height: 
        nheight = int(round((20.0 / width * height), 0))
        if (nheight == 0):
            nheight = 1
        img = im.resize((20, nheight), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wtop = int(round(((28 - nheight) / 2), 0))
        newImage.paste(img, (4, wtop))
    else:
        nwidth = int(round((20.0 / height * width), 0))
        if (nwidth == 0):
            nwidth = 1
        img = im.resize((nwidth, 20), Image.ANTIALIAS).filter(ImageFilter.SHARPEN)
        wleft = int(round(((28 - nwidth) / 2), 0)) 
        newImage.paste(img, (wleft, 4))
    tv = list(newImage.getdata())
    tva = [(255 - x) * 1.0 / 255.0 for x in tv]
    x=np.reshape(tva,(28,28))
    x=x.reshape(-1,1,28,28)
    return x

if __name__== "__main__":
    X_train, y_train = load_dataset("data10/dataall.txt")
    print(type(y_train[0]))
    X_test, y_test=X_train,y_train
    print(X_train[0].shape)
    batch_size=10
    num_epochs=5
    input_var = T.tensor4('inputs')
    target_var = T.ivector('targets')
    print("Building model and compiling functions...")
    network = build_cnn(input_var)
    prediction = lasagne.layers.get_output(network,input_shape=None)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean()
    train_acc = T.mean(T.eq(T.argmax(prediction, axis=1), target_var),
                      dtype=theano.config.floatX)
    params = lasagne.layers.get_all_params(network, trainable=True)
    updates = lasagne.updates.nesterov_momentum(
            loss, params, learning_rate=0.01, momentum=0.9)
    test_prediction = lasagne.layers.get_output(network, deterministic=True,input_shape=None)
    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,
                                                            target_var)
    test_loss = test_loss.mean()
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),
                      dtype=theano.config.floatX)
    train_fn = theano.function([input_var, target_var], [loss, train_acc], updates=updates,allow_input_downcast=True)
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc])
    check_fn=theano.function([input_var],test_prediction)
    print("Starting training...")
    for epoch in range(num_epochs):
        train_err = 0
        train_acc = 0
        train_batches = 0
        start_time = time.time()
        for batch in iterate_minibatches(X_train, y_train, batch_size, shuffle=True):
            inputs, targets = batch
            print("Step : "+str(train_batches));
            err, acc = train_fn(inputs, targets)
            train_err += err
            train_acc += acc
            train_batches += 1
        val_err = 0
        val_acc = 0
        val_batches = 0
    newData=load_testdata("data10/testdataConv.txt")
    output=load_testoutput("data10/datatesto.txt")
    test=len(newData)
    count=0
    for i in range(test):
        data=newData[i]
        x=str(np.argmax(max(check_fn([data]))))
        y=str(output[i])
        #plt.show(plt.imshow(x[0][0]))
        print("Input : "+y+" "+x)
        if(x==y):
            count=count+1
    print("Number of Samples : "+str(test))
    print("Number of Correct Outputs : "+str(count))

