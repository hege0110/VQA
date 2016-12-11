import numpy as np
from keras.utils import np_utils
import cv2, numpy as np
import pdb

def get_questions_tensor_timeseries(questions, nlp, timesteps):
	'''
	Returns a time series of word vectors for tokens in the question

	Input:
	questions: list of unicode objects
	nlp: an instance of the class English() from spacy.en
	timesteps: the number of 

	Output:
	A numpy ndarray of shape: (nb_samples, timesteps, word_vec_dim)
	'''
	assert not isinstance(questions, basestring)
	nb_samples = len(questions)
	word_vec_dim = nlp(questions[0])[0].vector.shape[0]
	questions_tensor = np.zeros((nb_samples, timesteps, word_vec_dim))
	for i in xrange(len(questions)):
		tokens = nlp(questions[i])
		for j in xrange(len(tokens)):
			if j<timesteps:
				questions_tensor[i,j,:] = tokens[j].vector

	return questions_tensor

def get_questions_matrix_sum(questions, nlp):
	'''
	Sums the word vectors of all the tokens in a question
	
	Input:
	questions: list of unicode objects
	nlp: an instance of the class English() from spacy.en

	Output:
	A numpy array of shape: (nb_samples, word_vec_dim)	
	'''
	assert not isinstance(questions, basestring)
	nb_samples = len(questions)
	word_vec_dim = nlp(questions[0])[0].vector.shape[0]
	questions_matrix = np.zeros((nb_samples, word_vec_dim))
	for i in xrange(len(questions)):
		tokens = nlp(questions[i])
		for j in xrange(len(tokens)):
			questions_matrix[i,:] += tokens[j].vector

	return questions_matrix

def get_answers_matrix(answers, encoder):
	'''
	Converts string objects to class labels

	Input:
	answers:	a list of unicode objects
	encoder:	a scikit-learn LabelEncoder object

	Output:
	A numpy array of shape (nb_samples, nb_classes)
	'''
	assert not isinstance(answers, basestring)
	y = encoder.transform(answers) #string to numerical class
	nb_classes = encoder.classes_.shape[0]
	Y = np_utils.to_categorical(y, nb_classes)
	return Y

def get_images_matrix(img_coco_ids, img_map, VGGfeatures):
	'''
	Gets the 4096-dimensional CNN features for the given COCO
	images
	
	Input:
	img_coco_ids: 	A list of strings, each string corresponding to
				  	the MS COCO Id of the relevant image
	img_map: 		A dictionary that maps the COCO Ids to their indexes 
					in the pre-computed VGG features matrix
	VGGfeatures: 	A numpy array of shape (nb_dimensions,nb_images)

	Ouput:
	A numpy matrix of size (nb_samples, nb_dimensions)
	'''
	assert not isinstance(img_coco_ids, basestring)
	nb_samples = len(img_coco_ids)
	nb_dimensions = VGGfeatures.shape[0]
	image_matrix = np.zeros((nb_samples, nb_dimensions))
	for j in xrange(len(img_coco_ids)):
		image_matrix[j,:] = VGGfeatures[:,img_map[img_coco_ids[j]]]

	return image_matrix

# def get_images_matrix(img_coco_ids, img_map, VGGfeatures):
# 	'''
# 	Gets the 4096-dimensional CNN features for the given COCO
# 	images
	
# 	Input:
# 	img_coco_ids: 	A list of strings, each string corresponding to
# 				  	the MS COCO Id of the relevant image
# 	img_map: 		A dictionary that maps the COCO Ids to their indexes 
# 					in the pre-computed VGG features matrix

# 	Ouput:
# 	A numpy matrix of size (nb_samples, nb_dimensions)
# 	'''
# 	assert not isinstance(img_coco_ids, basestring)
# 	nb_samples = len(img_coco_ids)
# 	nb_dimensions = VGGfeatures.shape[0]
# 	image_matrix = np.zeros((nb_samples, nb_dimensions))
# 	for j in xrange(len(img_coco_ids)):
# 		image_matrix[j,:] = VGGfeatures[:,img_map[img_coco_ids[j]]]

# 	return image_matrix

def get_images_matrix(img_coco_ids, VGGfeatures, VGGfeatures_reverse):
	'''
	Gets the 4096-dimensional CNN features for the given COCO
	images
	
	Input:
	img_coco_ids: 	A list of strings, each string corresponding to
				  	the MS COCO Id of the relevant image
	VGGfeatures: 	image features
	VGGfeatures_reverse: image features, stored in reverse order

	Ouput:
	A numpy matrix of size (nb_samples, nb_dimensions)
	'''
	assert not isinstance(img_coco_ids, basestring)
	nb_samples = len(img_coco_ids)
	nb_dimensions = VGGfeatures.shape[1]
	image_matrix = np.zeros((nb_samples, nb_dimensions))
	threshold1 = VGGfeatures.shape[0]
	threshold2 = 20000 - VGGfeatures_reverse.shape[0]
	for j in xrange(len(img_coco_ids)):
		index = int(img_coco_ids[j])
		if index < threshold1:
			image_matrix[j,:] = VGGfeatures[index]
		elif index >= threshold2:
			# 20000 -> 0
			# 12507 -> 20000 - 12507 = 7493
			image_matrix[j,:] = VGGfeatures_reverse[num_rev_rows-index]

	return image_matrix

def prepare_image(im_name,im_dir):
	im_name = im_name.zfill(12)
	im_path = im_dir + im_name +'.png'
	#pdb.set_trace()
	im = cv2.resize(cv2.imread(im_path), (224, 224)).astype(np.float32)
	im[:,:,0] -= 103.939
	im[:,:,1] -= 116.779
	im[:,:,2] -= 123.68
	im = im.transpose((2,0,1))
	im = np.expand_dims(im, axis=0)
	return im

def get_images_matrix_from_model(model, im_batch, im_dir, mapping):
	'''
	Gets the 4096-dimensional CNN features for the given COCO
	images

	Ouput:
	A numpy matrix of size (nb_samples, nb_dimensions)
	'''
	nb_samples = len(im_batch)
	nb_dimensions = 4096
	image_matrix = np.zeros((nb_samples, nb_dimensions))
	for j in xrange(nb_samples):
		if im_batch[j] in mapping:
			image_matrix[j,:] = mapping[im_batch[j]]
		else:
			im = prepare_image(im_batch[j],im_dir)
			temp = model.predict(im)
			image_matrix[j,:] = temp
			mapping[im_batch[j]] = temp
	return image_matrix