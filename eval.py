from data import *
import tensorflow as tf
from model import *
import os
import glob

# config
train_path = '/data/senseval3/eng-lex/EnglishLS.train.mod'
test_path = '/data/senseval3/eng-lex/EnglishLS.test.mod'

# load data
train_data = load_senteval3_data(train_path, is_training=True)
test_data = load_senteval3_data(test_path, is_training=False)
print 'Dataset size (train/test): %d / %d' % (len(train_data), len(test_data))

# build vocab utils
word_to_id = build_vocab(train_data)
target_word_to_id, target_sense_to_id, n_words, n_senses_from_target_id = build_sense_ids(train_data)
print 'Vocabulary size: %d' % len(word_to_id)

# make numeric
train_ndata = convert_to_numeric(train_data, word_to_id, target_word_to_id, target_sense_to_id, n_senses_from_target_id)
test_ndata = convert_to_numeric(test_data, word_to_id, target_word_to_id, target_sense_to_id, n_senses_from_target_id)

lexelts = get_lexelts(train_path)
target_word_to_lexelt = target_to_lexelt_map(target_word_to_id.keys(), lexelts)
# target_word_to_lexelt['colorless'] = target_word_to_lexelt['colourless']

target_id_to_word = {id: word for (word, id) in target_word_to_id.iteritems()}
target_id_to_sense_id_to_sense = [{sense_id: sense for (sense, sense_id) in sense_to_id.iteritems()} for (target_id, sense_to_id) in enumerate(target_sense_to_id)]

conf = {
    'batch_size': 100,
    'n_step_f': 147,
    'n_step_b': 42,
    'n_lstm_units': 74,
    'n_layers': 1,
    'emb_base_std': 0.10988,
    'input_keep_prob': .594688,
    'keep_prob': 0.193,
    'embedding_size': 100,
    'train_embeddings': True
}

with tf.Session() as session:
    print 'Restoring model'
    ckpt = tf.train.get_checkpoint_state('/home/salomons/tmp/model')

    with tf.variable_scope('model'):
        model = Model(False, conf, n_senses_from_target_id, word_to_id, None)

    saver = tf.train.Saver()
    saver.restore(session, '/home/salomons/tmp/model/wsd.ckpt-80')


    class Answer:
        pass

    print 'Evaluating'
    result = []
    for batch in batch_generator(conf['batch_size'], test_ndata, word_to_id['<pad>'], conf['n_step_f'], conf['n_step_b']):
        xfs, xbs, target_ids, sense_ids, instance_ids = batch
        feed = {
            model.inputs_f: xfs,
            model.inputs_b: xbs,
            model.train_target_ids: target_ids,
            model.train_sense_ids: sense_ids
        }

        predictions = session.run(model.predictions, feed_dict=feed)

        for i, predicted_sense_id in enumerate(predictions):
            a = Answer()
            a.target_word = target_id_to_word[target_ids[i]]
            a.lexelt = target_word_to_lexelt[a.target_word]
            a.instance_id = instance_ids[i]
            a.predicted_sense = target_id_to_sense_id_to_sense[target_ids[i]][predicted_sense_id]
            result.append(a)

    print 'Writing to file'
    path = '/home/salomons/tmp/result'
    with open(path, 'w') as file:
        for a in result:
            first = a.lexelt
            file.write('%s %s %s\n' % (first, a.instance_id, a.predicted_sense))
