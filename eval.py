from data import *
import tensorflow as tf
import model
import os
import glob

# config
train_path = '/data/senseval2/eng-lex-sample.training.xml'
test_path = '/data/senseval2/eng-lex-samp.evaluation.xml'

# load data
train_data = load_senteval2_data(train_path, is_training=True)
test_data = load_senteval2_data(test_path, is_training=False)
print 'Dataset size (train/test): %d / %d' % (len(train_data), len(test_data))

# build vocab utils
word_to_id = build_vocab(train_data)
target_word_to_id, target_sense_to_id, n_words, n_senses_from_target_id = build_sense_ids(train_data)
print 'Vocabulary size: %d' % len(word_to_id)

# make numeric
train_ndata = convert_to_numeric(train_data, word_to_id, target_word_to_id, target_sense_to_id, n_senses_from_target_id)
test_ndata = convert_to_numeric(test_data, word_to_id, target_word_to_id, target_sense_to_id, n_senses_from_target_id)

# calc max sentence length forwards and backwards
n_step_f = 40
n_step_b = 40
print 'n_step forward/backward: %d / %d' % (n_step_f, n_step_b)

lexelts = get_lexelts(train_path)
target_word_to_lexelt = target_to_lexelt_map(target_word_to_id.keys(), lexelts)

session = tf.Session()
saver = tf.train.Saver()

saver.restore(session, '/home/salomons/tmp/model/wsd1')

model = model.Model(False, 1, n_step_f, n_step_b, None)

for batch in batch_generator(1, train_ndata, target_word_to_id('<pad>'), n_step_f, n_step_b):
    pass



