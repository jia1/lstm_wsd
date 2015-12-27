from data import *
from glove import *
import tensorflow as tf
from sklearn.cross_validation import train_test_split
import os
import glob
from model import *

# rm old log files
for file in glob.glob('/home/salomons/tmp/tf.log/*'):
    os.remove(file)

# config
se_2_or_3 = 2
validate = True
n_epochs = 100
conf = {
    'batch_size': 100,
    'n_step_f': 100,
    'n_step_b': 42,
    'n_lstm_units': 74,
    'n_layers': 1,
    'emb_base_std': 0.20988,
    'input_keep_prob': 0.5,
    'keep_prob': 0.5,
    'embedding_size': 100,
    'train_embeddings': True,
    'forget_bias': 0.0,
    'state_size': 200
}

# load data
train_data = load_train_data(se_2_or_3)
test_data = load_test_data(se_2_or_3)
print 'Dataset size (train/test): %d / %d' % (len(train_data), len(test_data))

# build vocab utils
word_to_id = build_vocab(train_data)
target_word_to_id, target_sense_to_id, n_words, n_senses_from_target_id = build_sense_ids(train_data)
print 'Vocabulary size: %d' % len(word_to_id)

# make numeric
train_ndata = convert_to_numeric(train_data, word_to_id, target_word_to_id, target_sense_to_id, n_senses_from_target_id)
test_ndata = convert_to_numeric(test_data, word_to_id, target_word_to_id, target_sense_to_id, n_senses_from_target_id)




test_size = 0.2 if validate else 0.0
train_data, val_data = train_test_split(train_ndata, test_size=test_size)

init_emb = fill_with_gloves(word_to_id, 100)

with tf.variable_scope('model', reuse=None):
    model_train = Model(True, conf, n_senses_from_target_id, word_to_id, init_emb)
with tf.variable_scope('model', reuse=True):
    model_val = Model(False, conf, n_senses_from_target_id, word_to_id, init_emb)

saver = tf.train.Saver(tf.all_variables(), max_to_keep=100)

session = tf.Session()
session.run(tf.initialize_all_variables())

# writer = tf.train.SummaryWriter('/home/salomons/tmp/tf.log', session.graph_def, flush_secs=10)

for i in range(n_epochs):
    print '::: EPOCH: %d :::' % i

    summaries = run_epoch(session, model_train, conf, train_data, 'train', word_to_id)
    if validate:
        run_epoch(session, model_val, conf, val_data, 'val', word_to_id)

    # for batch_idx, summary in enumerate(summaries):
    #     writer.add_summary(summary, i*len(train_data)//batch_size + batch_idx)

    if i % 5 == 0:
        print saver.save(session, '/home/salomons/tmp/model/wsd.ckpt', global_step=i)
