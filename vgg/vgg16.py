from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.utils import generic_utils
import cv2, numpy as np
import pdb
import h5py
import os

def VGG_16(weights_path):
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(3,224,224)))
    model.add(Convolution2D(64, 3, 3, activation='relu' ))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, activation='relu' ))
    model.add(MaxPooling2D((2,2), stride=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu' ))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3, activation='relu' ))
    model.add(MaxPooling2D((2,2), stride=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu' ))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu' ))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3, activation='relu' ))
    model.add(MaxPooling2D((2,2), stride=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu' ))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu' ))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu' ))
    model.add(MaxPooling2D((2,2), stride=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu' ))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu' ))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3, activation='relu' ))
    model.add(MaxPooling2D((2,2), stride=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(4096, activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Dense(1000, activation='softmax'))

    # if weights_path:
    #     model.load_weights(weights_path)
    f = h5py.File(weights_path)
    for k in range(f.attrs['nb_layers']):
        if k >= len(model.layers):
            # we don't look at the last (fully-connected) layers in the savefile
            break
        g = f['layer_{}'.format(k)]
        weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
        model.layers[k].set_weights(weights)
    f.close()
    print('Weights loaded!')


    return model

# def load_weights(model, weights_path):

def get_prediction(img_path, model, f):
    # print img_path
    im = cv2.resize(cv2.imread(img_path), (224, 224)).astype(np.float32)
    im[:,:,0] -= 103.939
    im[:,:,1] -= 116.779
    im[:,:,2] -= 123.68
    im = im.transpose((2,0,1))
    im = np.expand_dims(im, axis=0)
    # return model.predict(im)[0]
    out = model.predict(im)
    np.savetxt(f, out)

if __name__ == "__main__":
    # im = cv2.resize(cv2.imread('cat.jpg'), (224, 224)).astype(np.float32)
    # im[:,:,0] -= 103.939
    # im[:,:,1] -= 116.779
    # im[:,:,2] -= 123.68
    # im = im.transpose((2,0,1))
    # im = np.expand_dims(im, axis=0)
    cwd = os.getcwd()
    f = open(cwd + '/abstract_image_precompute.txt', 'w+')
    num_images = 20000
    progbar = generic_utils.Progbar(num_images)

    # Test pretrained model
    model = VGG_16('vgg16_weights.h5')
    #model = load_weights(model, 'vgg16_weights.h5')
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    # f = open('abstract_image_precompute.txt')

    for i in range(num_images-1, -1, -1):
        img_path = 'scene_img_abstract_v002_train2015/abstract_v002_train2015_' + "%0*d" % (12, i) + '.png'
        get_prediction(img_path, model, f)
        progbar.add(1)
