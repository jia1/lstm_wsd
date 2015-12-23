from data import *
from glove import *
import tensorflow as tf
import tensorflow.models.rnn.rnn_cell as rnn_cell
from sklearn.cross_validation import train_test_split
import os
import glob
from model import *

# rm old log files
for file in glob.glob('/home/salomons/tmp/tf.log/*'):
    os.remove(file)

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


def debug(model, session, feed_dict):
    for name, op in model.dbg.iteritems():
        value = session.run(op, feed_dict)
        print '::: %s :::: \n%s' % (name, np.array_str(value))


def debug_op(op, session, feed_dict):
    value = session.run(op, feed_dict)
    print value

n_epochs = 300
conf = {
    'batch_size': 100,
    'n_step_f': 80,
    'n_step_b': 80,
    'n_lstm_units': 20,
    'n_layers': 1,
    'emb_base_std': 0.5,
    'input_keep_prob': 1.0,
    'keep_prob': 0.5,
    'embedding_size': 100
}

train_data, val_data = train_test_split(train_ndata, test_size=0.2)

init_emb = fill_with_gloves(word_to_id, 100)

with tf.variable_scope('model', reuse=None):
    model_train = Model(True, conf, init_emb)
with tf.variable_scope('model', reuse=True):
    model_val = Model(False, conf, init_emb)

saver = tf.train.Saver(tf.all_variables(), max_to_keep=100)

tf_config = tf.ConfigProto(inter_op_parallelism_threads=4,
                           intra_op_parallelism_threads=4)

session = tf.Session(config=tf_config)
session.run(tf.initialize_all_variables())

# writer = tf.train.SummaryWriter('/home/salomons/tmp/tf.log', session.graph_def, flush_secs=10)

for i in range(n_epochs):
    print '::: EPOCH: %d :::' % i

    summaries = run_epoch(session, model_train, conf, train_data, 'train', i)
    run_epoch(session, model_val, conf, val_data, 'val', i)

    # for batch_idx, summary in enumerate(summaries):
    #     writer.add_summary(summary, i*len(train_data)//batch_size + batch_idx)

    if i % 1 == 0:
        print saver.save(session, '/home/salomons/tmp/model/wsd.ckpt', global_step=i)
