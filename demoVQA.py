import sys
import os
from random import shuffle
import argparse

import numpy as np
import scipy.io

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.utils import np_utils, generic_utils

from sklearn import preprocessing
from sklearn.externals import joblib

from spacy.en import English

from features import get_questions_matrix_sum, get_images_matrix, get_answers_matrix,get_images_matrix_from_model
from utils import grouper, selectFrequentAnswers

from vgg import vgg16
#abstract_v002_train2015_000000000004.png
def main():
	cwd = os.getcwd()

	parser = argparse.ArgumentParser()
	parser.add_argument('-num_hidden_units', type=int, default=1024)
	parser.add_argument('-num_hidden_layers', type=int, default=3)
	parser.add_argument('-dropout', type=float, default=0.5)
	parser.add_argument('-activation', type=str, default='tanh')
	parser.add_argument('-language_only', type=bool, default= False)
	parser.add_argument('-num_epochs', type=int, default=2)
	parser.add_argument('-model_save_interval', type=int, default=10)
	parser.add_argument('-model_weights_path', type=str, default=cwd+'/vgg/vgg16_weights.h5')
	parser.add_argument('-batch_size', type=int, default=128)
	parser.add_argument('-questions_train',type=str, default = cwd+'/data/preprocessed/questions_train2015.txt')
	parser.add_argument('-answers_train',type=str, default = cwd+'/data/preprocessed/answers_train2015_modal.txt')
	parser.add_argument('-im_dir',type=str, default =cwd+'/data/preprocessed/scene_img_abstract_v002_train2015/')
	#parser.add_argument('-questions_train',type=str, default = cwd+'/data/preprocessed/questions_train2014.txt')
	args = parser.parse_args()

	questions_train = open(args.questions_train, 'r').read().decode('utf8').splitlines()
	answers_train = open(args.answers_train, 'r').read().decode('utf8').splitlines()
	images_train = open(cwd+'/data/preprocessed/images_train2015.txt', 'r').read().decode('utf8').splitlines()
	#vgg_model_path = cwd+'/features/coco/vgg_feats.mat' #this needs to change
	maxAnswers = 100
	questions_train, answers_train, images_train = selectFrequentAnswers(questions_train,answers_train,images_train, maxAnswers)

	#encode the remaining answers
	labelencoder = preprocessing.LabelEncoder()
	labelencoder.fit(answers_train)
	nb_classes = len(list(labelencoder.classes_))
	joblib.dump(labelencoder,cwd+'/models/labelencoder.pkl')

	#features_struct = scipy.io.loadmat(vgg_model_path)
	#VGGfeatures = features_struct['feats']
	# print 'loaded vgg features'
	# image_ids = open(cwd+'/features/coco_vgg_IDMap.txt').read().splitlines()
	# id_map = {}
	# for ids in image_ids:
	# 	id_split = ids.split()
	# 	id_map[id_split[0]] = int(id_split[1])

	vgg_model = vgg16.VGG_16(args.model_weights_path)
	sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
	vgg_model.compile(optimizer=sgd, loss='categorical_crossentropy')
	print 'loaded vgg model...'

	nlp = English()
	print 'loaded word2vec features...'

	img_dim = 4096
	word_vec_dim = 300

	model = Sequential()
	if args.language_only:
		model.add(Dense(args.num_hidden_units, input_dim=word_vec_dim, init='uniform'))
	else:
		model.add(Dense(args.num_hidden_units, input_dim=img_dim+word_vec_dim, init='uniform'))
	model.add(Activation(args.activation))
	if args.dropout>0:
		model.add(Dropout(args.dropout))
	for i in xrange(args.num_hidden_layers-1):
		model.add(Dense(args.num_hidden_units, init='uniform'))
		model.add(Activation(args.activation))
		if args.dropout>0:
			model.add(Dropout(args.dropout))
	model.add(Dense(nb_classes, init='uniform'))
	model.add(Activation('softmax'))

	json_string = model.to_json()
	model_file_name = cwd+'/models/mlp_num_hidden_units_' + str(args.num_hidden_units) + '_num_hidden_layers_' + str(args.num_hidden_layers)		
	open(model_file_name  + '.json', 'w').write(json_string)
	
	print 'Training started...'
	id_map = {}
	for k in xrange(args.num_epochs):
		#shuffle the data points before going through them
		index_shuf = range(len(questions_train))
		shuffle(index_shuf)
		questions_train = [questions_train[i] for i in index_shuf]
		answers_train = [answers_train[i] for i in index_shuf]
		images_train = [images_train[i] for i in index_shuf]
		progbar = generic_utils.Progbar(len(questions_train))
		for qu_batch,an_batch,im_batch in zip(grouper(questions_train, args.batch_size, fillvalue=questions_train[-1]), 
											grouper(answers_train, args.batch_size, fillvalue=answers_train[-1]), 
											grouper(images_train, args.batch_size, fillvalue=images_train[-1])):
			
			X_q_batch = get_questions_matrix_sum(qu_batch, nlp)
			im_path = args.im_dir +"abstract_v002_train2015_"
			print 'getting image features...'
			X_i_batch = get_images_matrix_from_model(vgg_model, im_batch, im_path, id_map)
			X_batch = np.hstack((X_q_batch, X_i_batch))

			Y_batch = get_answers_matrix(an_batch, labelencoder)
			print 'running training on batch...'
			loss = model.train_on_batch(X_batch, Y_batch)

			progbar.add(args.batch_size, values=[("train loss", loss)])

		if k%args.model_save_interval == 0:
			model.save_weights(model_file_name + '_epoch_{:02d}.hdf5'.format(k))

	model.save_weights(model_file_name + '_epoch_{:02d}.hdf5'.format(k))

if __name__ == "__main__":
	main()