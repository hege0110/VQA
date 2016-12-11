import argparse
import random
from PIL import Image
import subprocess
from os import listdir
from os.path import isfile, join
import os
from keras.models import model_from_json

from spacy.en import English
import numpy as np
import scipy.io
from sklearn.externals import joblib
import pdb
from features import get_questions_tensor_timeseries, get_images_matrix

def main():
  cwd = os.getcwd()

  parser = argparse.ArgumentParser()
  parser.add_argument('--model', type=str, default=cwd+'/models/lstm_1_num_hidden_units_lstm_512_num_hidden_units_mlp_1024_num_hidden_layers_mlp_3.json')
  parser.add_argument('--weights', type=str, default=cwd+'/models/lstm_1_num_hidden_units_lstm_512_num_hidden_units_mlp_1024_num_hidden_layers_mlp_3_epoch_070.hdf5')
  parser.add_argument('--sample_size', type=int, default=25)
  parser.add_argument('--caffe', help='path to caffe installation')
  parser.add_argument('--model_def', help='path to model definition prototxt')
  parser.add_argument('--vggmodel', default='VGG_ILSVRC_16_layers.caffemodel', help='path to model parameters')
  args = parser.parse_args()

  print 'Loading word2vec'
  nlp = English()
  print 'Loaded word2vec features'
  labelencoder = joblib.load(cwd+'/models/labelencoder.pkl')
  print 'Loading Model'
  args.model = cwd+"/"+args.model
  #pdb.set_trace()
  model = model_from_json(open(args.model).read())
  print 'Loading Weights'
  args.weights = cwd+"/"+args.weights
  model.load_weights(args.weights)
  model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
  print 'Loaded'
  q = True

  while q:

     path = str(raw_input('Enter path to image : '))
     if path != 'same':
         base_dir = os.path.dirname(path)
         os.system('python extract_features.py --caffe ' + str(args.caffe) + ' --model_def vgg_features.prototxt --gpu --model ' + str(args.vggmodel) + ' --image ' + path )
     print 'Loading VGGfeats'
     vgg_model_path = os.path.join(base_dir + '/vgg_feats.mat')
     features_struct = scipy.io.loadmat(vgg_model_path)
     VGGfeatures = features_struct['feats']
     print "Loaded"

     question = unicode(raw_input("Ask a question: "))
     if question == "quit":
         q = False
     timesteps = len(nlp(question))
     X_q = get_questions_tensor_timeseries([question], nlp, timesteps)
     X_i = np.reshape(VGGfeatures, (1, 4096))

     X = [X_q, X_i]

     y_predict = model.predict_classes(X, verbose=0)
     print labelencoder.inverse_transform(y_predict)


main()