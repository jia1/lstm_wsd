from data import *
from glove import *
import tensorflow as tf
import tensorflow.models.rnn.rnn_cell as rnn_cell
from sklearn.cross_validation import train_test_split
import os
import glob

# rm old log files
for file in glob.glob('/home/salomons/tmp/tf.log/*'):
    os.remove(file)

# config
train_path = '/data/senseval2/eng-lex-sample.training.xml'
test_path = '/data/senseval2/eng-lex-samp.evaluation.xml'

# load data
train_data = load_senteval2_data(train_path)
test_data = load_senteval2_data(test_path)
print 'Dataset size (train/test): %d / %d' % (len(train_data), len(test_data))

# build vocab utils
word_to_id = build_vocab(train_data)
target_word_to_id, target_sense_to_id, n_words, n_senses_from_target_id = build_sense_ids(train_data)
print 'Vocabulary size: %d' % len(word_to_id)

# make numeric
train_ndata = convert_to_numeric(train_data, word_to_id, target_word_to_id, target_sense_to_id, n_senses_from_target_id)
test_ndata = convert_to_numeric(test_data, word_to_id, target_word_to_id, target_sense_to_id, n_senses_from_target_id)

n_step_f = 40
n_step_b = 40
print 'n_step forward/backward: %d / %d' % (n_step_f, n_step_b)


class Model:
    def __init__(self, is_first, target_id, n_step_f, n_step_b, init_word_vecs=None):
        self.inputs_f = tf.placeholder(tf.int32, shape=[None, n_step_f])
        self.inputs_b = tf.placeholder(tf.int32, shape=[None, n_step_b])
        self.target_ids = tf.placeholder(tf.int32, shape=[None])
        self.sense_ids = tf.placeholder(tf.int32, shape=[None])

        n_units = 100
        state_size = 2 * n_units
        n_senses = n_senses_from_target_id[target_id]

        reuse = None if is_first else True

        vocab_size = len(word_to_id)
        embedding_size = 100
        def embedding_initializer(vec, dtype):
            return init_word_vecs if init_word_vecs else tf.random_uniform([vocab_size, embedding_size], -.1, .1, dtype)

        with tf.variable_scope('emb', reuse):
            embeddings = tf.get_variable('embeddings', [vocab_size, embedding_size], initializer=embedding_initializer)

        with tf.variable_scope(str(target_id)):
            W_target = tf.get_variable('W_target', [state_size*2, n_senses], tf.float32)
            b_target = tf.get_variable('b_target', [1, n_senses])

        keep_prop = 0.5
        with tf.variable_scope("forward", reuse):
            f_lstm = rnn_cell.DropoutWrapper(rnn_cell.BasicLSTMCell(n_units), output_keep_prob=keep_prop)
            f_state = tf.get_variable('f_state', [None, state_size], initializer=tf.constant_initializer(0.0), trainable=False)

            # run inputs through lstm
            inputs_f = tf.split(1, n_step_f, self.inputs_f)
            for time_step, inputs_ in enumerate(inputs_f):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                emb = tf.nn.embedding_lookup(embeddings, tf.squeeze(inputs_))
                _, f_state = f_lstm(emb, f_state)

        with tf.variable_scope("backward", reuse):
            b_lstm = rnn_cell.DropoutWrapper(rnn_cell.BasicLSTMCell(n_units), output_keep_prob=keep_prop)
            # b_state = b_lstm.zero_state(None, tf.float32)
            b_state = tf.Variable(tf.zeros([None, state_size]))

            inputs_b = tf.split(1, n_step_b, self.inputs_b)
            for time_step, inputs_ in enumerate(inputs_b):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                emb = tf.nn.embedding_lookup(embeddings, tf.squeeze(inputs_))
                _, b_state = b_lstm(emb, b_state)

        concat_state = tf.concat(1, [f_state, b_state])
        state = tf.nn.dropout(concat_state, keep_prop)

        logits = tf.matmul(state, W_target) + b_target

        self.cost_op = tf.nn.softmax_cross_entropy_with_logits(logits, self.sense_ids)
        self.accuracy_op = tf.reduce_mean(tf.cast( tf.equal(tf.arg_max(logits, 1), self.sense_ids), tf.float32))

        grads = tf.gradients(self.cost_op, W_target)
        for grad in grads:
            print tf.shape(grad)
        tf.histogram_summary('grad_W_target', grads[0])
        tf.scalar_summary('frac_0_grad_W', tf.nn.zero_fraction(grads[0]))

        print 'TRAINABLE VARIABLES'
        tvars = tf.trainable_variables()
        for tvar in tvars:
            print tvar.name

        # max_grad_norm = 10
        # grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost_op, tvars), max_grad_norm)
        # optimizer = tf.train.AdagradOptimizer(.5)
        # self.train_op = optimizer.apply_gradients(zip(grads, tvars))
        self.train_op = tf.train.AdagradOptimizer(0.2).minimize(self.cost_op)
        self.summary_op = tf.merge_all_summaries()


def run_epoch(session, models, train_data, val_data):
    train_cost = 0.
    train_acc = 0.
    val_cost = 0.
    val_acc = 0.

    summaries = []
    n_train_batches = 0
    for target_id, data in train_data.iteritems():
        model = models[target_id]
        xf, xb, target_ids, sense_ids = data

        batch_cost, batch_acc, summary, _ = session.run([model.cost_op, model.accuracy_op, model.summary_op, model.train_op], {
            model.inputs_f: xf,
            model.inputs_b: xb,
            model.target_ids: target_ids,
            model.sense_ids: sense_ids
        })

        train_cost += batch_cost
        train_acc += batch_acc
        summaries.append(summary)
        n_train_batches += 1

    n_val_batches = 0
    for target_id, data in val_data.iteritems():
        xf, xb, target_ids, sense_ids = data

        batch_cost, batch_acc = session.run([model.cost_op, model.accuracy_op], {
            model.inputs_f: xf,
            model.inputs_b: xb,
            model.target_ids: target_ids,
            model.sense_ids: sense_ids
        })

        val_cost += batch_cost
        val_acc += batch_acc
        n_val_batches += 1

    print 'Cost (train/val): %f/%f, Accuracy (train/val): %f/%f' \
          % (train_cost / n_train_batches, val_cost / n_val_batches, train_acc / n_train_batches, val_acc / n_val_batches )

    return summaries


if __name__ == '__main__':
    n_epochs = 500

    grouped_by_target = group_by_target(train_ndata)
    train_data, val_data = split_grouped(grouped_by_target, .2, 2)

    init_emb = fill_with_gloves(word_to_id, 100)

    models = {}
    is_first = True
    for target_id in grouped_by_target.keys():
        models[target_id] = Model(is_first, target_id, n_step_f, n_step_b, init_emb)
        is_first = False

    session = tf.Session()
    session.run(tf.initialize_all_variables())

    summary_op = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter('/home/salomons/tmp/tf.log', session.graph_def, flush_secs=10)

    for i in range(n_epochs):
        print 'EPOCH: %d' % i
        summaries = run_epoch(session, models, train_data, val_data)
        for batch_idx, summary in enumerate(summaries):
            writer.add_summary(summary)

