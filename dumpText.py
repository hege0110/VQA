import operator
import argparse
import progressbar
import json
from spacy.en import English
import os
import commands
def getModalAnswer(answers):
	candidates = {}
	for i in xrange(10):
		candidates[answers[i]['answer']] = 1

	for i in xrange(10):
		candidates[answers[i]['answer']] += 1

	return max(candidates.iteritems(), key=operator.itemgetter(1))[0]

def getAllAnswer(answers):
	answer_list = []
	for i in xrange(10):
		answer_list.append(answers[i]['answer'])

	return ';'.join(answer_list)

def main():
	cwd = os.getcwd()
	print(cwd)
	parser = argparse.ArgumentParser()
	parser.add_argument('-split', type=str, default='train', 
		help='Specify which part of the dataset you want to dump to text. Your options are: train, val, test, test-dev')
	parser.add_argument('-answers', type=str, default='modal',
		help='Specify if you want to dump just the most frequent answer for each questions (modal), or all the answers (all)')
	args = parser.parse_args()

	nlp = English() #used for conting number of tokens

	if args.split == 'train':
		annFile = cwd + "/data/abstract_v002_train2015_annotations.json"
		quesFile = cwd + "/data/MultipleChoice_abstract_v002_train2015_questions.json"
		questions_file = open(cwd + "/data/preprocessed/questions_train2015.txt", 'w+')
		questions_id_file = open(cwd + "/data/preprocessed/questions_id_train2015.txt", 'w+')
		questions_lengths_file = open(cwd + "/data/preprocessed/questions_lengths_train2015.txt", 'w+')
		if args.answers == 'modal':
			answers_file = open(cwd + "/data/preprocessed/answers_train2015_modal.txt", 'w+')
		elif args.answers == 'all':
			answers_file = open(cwd + "/data/preprocessed/answers_train2015_all.txt", 'w+')
		coco_image_id = open(cwd + "/data/preprocessed/images_train2015.txt", 'w+')
		data_split = 'training data'
	elif args.split == 'val':
		annFile = cwd + "/data/abstract_v002_val2015_annotations.json"
		quesFile = cwd + "/data/MultipleChoice_abstract_v002_val2015_questions.json"
		questions_file = open(cwd + "/data/preprocessed/questions_val2015.txt", 'w+')
		questions_id_file = open(cwd + "/data/preprocessed/questions_id_val2015.txt", 'w+')
		questions_lengths_file = open(cwd + "/data/preprocessed/questions_lengths_val2015.txt", 'w+')
		if args.answers == 'modal':
			answers_file = open(cwd + "/data/preprocessed/answers_val2015_modal.txt", 'w+')
		elif args.answers == 'all':
			answers_file = open(cwd + "/data/preprocessed/answers_val2015_all.txt", 'w+')
		coco_image_id = open(cwd + "/data/preprocessed/images_val2015_all.txt", 'w+')
		data_split = 'validation data'
	elif args.split == 'test-dev':
		quesFile = cwd + "/data/MultipleChoice_mscoco_test-dev2015_questions.json"
		questions_file = open(cwd + "/data/preprocessed/questions_test-dev2015.txt", 'w+')
		questions_id_file = open(cwd + "/data/preprocessed/questions_id_test-dev2015.txt", 'w+')
		questions_lengths_file = open(cwd + "/data/preprocessed/questions_lengths_test-dev2015.txt", 'w+')
		coco_image_id = open(cwd + "/data/preprocessed/images_test-dev2015.txt", 'w+')
		data_split = 'test-dev data'
	elif args.split == 'test':
		quesFile = cwd + "/data/MultipleChoice_abstract_v002_test2015_questions.json"
		questions_file = open(cwd + "/data/preprocessed/questions_test2015.txt", 'w+')
		questions_id_file = open(cwd + "/data/preprocessed/questions_id_test2015.txt", 'w+')
		questions_lengths_file = open(cwd + "/data/preprocessed/questions_lengths_test2015.txt", 'w+')
		coco_image_id = open(cwd + "/data/preprocessed/images_test2015.txt", 'w+')
		data_split = 'test data'
	else:
		raise RuntimeError('Incorrect split. Your choices are:\ntrain\nval\ntest-dev\ntest')

	#initialize VQA api for QA annotations
	#vqa=VQA(annFile, quesFile)
	questions = json.load(open(quesFile, 'r'))
	ques = questions['questions']
	if args.split == 'train' or args.split == 'val':
		qa = json.load(open(annFile, 'r'))
		qa = qa['annotations']
	
	pbar = progressbar.ProgressBar()
	print 'Dumping questions, answers, questionIDs, imageIDs, and questions lengths to text files...'
	f1_len = commands.getstatusoutput('wc -l abstract_image_precompute.txt')
	f2_len = commands.getstatusoutput('wc -l abstract_image_precompute_reverse.txt')
	THRESHOLD1 = int(filter(None, f1_len[1].split(' '))[0])
	THRESHOLD2 = 20000 - int(filter(None, f2_len[1].split(' '))[0])
	for i, q in pbar(zip(xrange(len(ques)),ques)):
		# questions_file.write((q['question'] + '\n').encode('utf8'))
		# questions_lengths_file.write((str(len(nlp(q['question'])))+ '\n').encode('utf8'))
		if (q['question_id'] >= THRESHOLD1 and q['question_id'] < THRESHOLD2):
			continue
		question = q['question'] + ' ' + ' '.join(q['multiple_choices'])
		questions_file.write((question + '\n').encode('utf8'))
		questions_lengths_file.write((str(len(nlp(question)))+ '\n').encode('utf8'))
		questions_id_file.write((str(q['question_id']) + '\n').encode('utf8'))
		coco_image_id.write((str(q['image_id']) + '\n').encode('utf8'))
		if args.split == 'train' or args.split == 'val':
			if args.answers == 'modal':
				answers_file.write(getModalAnswer(qa[i]['answers']).encode('utf8'))
			elif args.answers == 'all':
				answers_file.write(getAllAnswer(qa[i]['answers']).encode('utf8'))
			answers_file.write('\n'.encode('utf8'))

	print 'completed dumping', data_split

if __name__ == "__main__":
	main()