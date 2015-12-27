#!/usr/bin/env python2.7
# encoding: utf-8

from data import *
import time
from sklearn.cross_validation import train_test_split
from glove import *
from model import *
import argparse


se_2_or_3 = 2


# /home/salomons/project/wsd/hyper.py --seed 7768176 --model-stdout --dimacs no_instance --tmout 1801 --n_layers 1 --n_lstm_units 20 --n_step_b 40 --n_step_f 40 --input_keep_prob 0.9 --emb_base_std 0.5 --batch_size 20 --embedding_size 100 --keep_prob 0.5

parser = argparse.ArgumentParser()
parser.add_argument('--seed')
parser.add_argument('--dimacs')
parser.add_argument('--tmout')
parser.add_argument('--model-stdout', action='store_true')

parser.add_argument('--batch_size')
parser.add_argument('--emb_base_std')
parser.add_argument('--embedding_size')
parser.add_argument('--input_keep_prob')
parser.add_argument('--keep_prob')
parser.add_argument('--n_layers')
parser.add_argument('--n_lstm_units')
parser.add_argument('--n_step_f')
parser.add_argument('--n_step_b')
parser.add_argument('--train_embeddings')
parser.add_argument('--forget_bias')
parser.add_argument('--state_size')

args = parser.parse_args()

conf = {
    'batch_size': int(args.batch_size),
    'n_step_f': int(args.n_step_f),
    'n_step_b': int(args.n_step_b),
    'n_lstm_units': int(args.n_lstm_units),
    'n_layers': int(args.n_layers),
    'emb_base_std': float(args.emb_base_std),
    'input_keep_prob': float(args.input_keep_prob),
    'keep_prob': float(args.keep_prob),
    'embedding_size': int(args.embedding_size),
    'train_embeddings': bool(args.train_embeddings),
    'forget_bias': float(args.forget_bias),
    'state_size': int(args.state_size)
}

start_time = time.time()

# config
n_epochs = 100
max_sec = 40 * 60
seed = int(args.seed)
tf.set_random_seed(seed)
np.random.seed(seed)

# load data
train_data = load_train_data(se_2_or_3)

# build vocab utils
word_to_id = build_vocab(train_data)
target_word_to_id, target_sense_to_id, n_words, n_senses_from_target_id = build_sense_ids(train_data)

# make numeric and split
train_ndata = convert_to_numeric(train_data, word_to_id, target_word_to_id, target_sense_to_id, n_senses_from_target_id)
train_data, val_data = train_test_split(train_ndata, test_size=0.2)

init_emb = fill_with_gloves(word_to_id, conf['embedding_size'])

with tf.variable_scope('model', reuse=None):
    model_train = Model(True, conf, n_senses_from_target_id, word_to_id, init_emb)
with tf.variable_scope('model', reuse=True):
    model_val = Model(False, conf, n_senses_from_target_id, word_to_id, init_emb)

session = tf.Session()
session.run(tf.initialize_all_variables())

best = 0.0
i_since_last_best = 0
max_i_wo_impr = 12
for i in range(n_epochs):
    print '::: EPOCH: %d :::' % i

    cost_train, acc_train = run_epoch(session, model_train, conf, train_data, 'train', word_to_id)
    cost_val, acc_val = run_epoch(session, model_val, conf, val_data, 'val', word_to_id)

    if acc_val > best:
        print 'NEW BEST: %f' % acc_val
        i_since_last_best = 0
        best = acc_val
    else:
        i_since_last_best += 1

    # max wo improvement
    if i_since_last_best > max_i_wo_impr:
        break

    # timeout
    if time.time() - start_time > max_sec:
        break

