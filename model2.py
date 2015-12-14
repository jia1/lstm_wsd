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



# calc max sentence length forwards and backwards
n_step_f = 40
n_step_b = 40
print 'n_step forward/backward: %d / %d' % (n_step_f, n_step_b)


class Model:
    def __init__(self, n_step_f, n_step_b, init_word_vecs=None):
        self.batch_size = batch_size

        self.train_inputs = train_inputs = tf.placeholder(tf.int32, shape=[None, n_step_f])
        self.train_target_ids = train_target_ids = tf.placeholder(tf.int32, shape=[batch_size])
        self.train_sense_ids = train_sense_ids = tf.placeholder(tf.int32, shape=[batch_size])

        tot_n_senses = sum(n_senses_from_target_id.values())
        self.train_labels = labels = tf.placeholder(tf.float32, shape=[batch_size, tot_n_senses])

        vocab_size = len(word_to_id)
        init_emb = init_word_vecs if init_word_vecs else tf.random_uniform([vocab_size, 100], -.1, .1)
        embeddings = tf.Variable(init_emb, name='embeddings')

        n_units = 100
        state_size = 2 * n_units

        print 'Avg n senses: ' + str(tot_n_senses / len(n_senses_from_target_id))

        n_senses_sorted_by_target_id = [n_senses_from_target_id[target_id] for target_id
                                        in range(len(n_senses_from_target_id))]
        n_senses_sorted_by_target_id_tf = tf.constant(n_senses_sorted_by_target_id, tf.int32)
        _W_starts = (np.cumsum(np.append([0], n_senses_sorted_by_target_id)) * 2 * state_size)[:-1]
        _W_lenghts = np.array(n_senses_sorted_by_target_id) * 2 * state_size
        W_starts = tf.constant(_W_starts, tf.int32)
        W_lengths = tf.constant(_W_lenghts, tf.int32)

        _b_starts = (np.cumsum(np.append([0], n_senses_sorted_by_target_id)))[:-1]
        _b_lengths = np.array(n_senses_sorted_by_target_id)
        b_starts = tf.constant(_b_starts, tf.int32)
        b_lengths = tf.constant(_b_lengths, tf.int32)

        W_target = tf.Variable(tf.random_uniform([tot_n_senses * 2 * state_size], -0.1, 0.1, dtype=tf.float32), name='W_target')
        b_target = tf.Variable(tf.zeros([tot_n_senses], dtype=tf.float32), name='b_target')

        keep_prop = 0.5
        with tf.variable_scope("forward"):
            f_lstm = rnn_cell.DropoutWrapper(rnn_cell.BasicLSTMCell(n_units), output_keep_prob=keep_prop)
            f_state = f_lstm.zero_state(batch_size, tf.float32)

            # run inputs through lstm
            inputs_f = tf.split(1, n_step_f, train_inputs)
            for time_step, inputs_ in enumerate(inputs_f):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                emb = tf.nn.embedding_lookup(embeddings, tf.squeeze(inputs_))
                _, f_state = f_lstm(emb, f_state)

        with tf.variable_scope("backward"):
            b_lstm = rnn_cell.DropoutWrapper(rnn_cell.BasicLSTMCell(n_units), output_keep_prob=keep_prop)
            b_state = b_lstm.zero_state(batch_size, tf.float32)

            inputs_b = tf.split(1, n_step_b, train_inputs)
            for time_step, inputs_ in enumerate(inputs_b):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                emb = tf.nn.embedding_lookup(embeddings, tf.squeeze(inputs_))
                _, b_state = b_lstm(emb, b_state)

        concat_state = tf.concat(1, [f_state, b_state])
        state = tf.nn.dropout(concat_state, keep_prop)

        loss = tf.Variable(0., trainable=False)
        n_correct = tf.Variable(0, trainable=False)

        unbatched_states = tf.split(0, batch_size, state)
        unbatched_target_ids = tf.split(0, batch_size, train_target_ids)
        unbatched_sense_ids = tf.split(0, batch_size, train_sense_ids)
        one = tf.constant(1, tf.int32, [1])

        # make predictions for all instances in batch
        for i in range(batch_size):
            target_id = unbatched_target_ids[i]  # tf.split(train_target_ids, i, [1])
            sense_id = unbatched_sense_ids[i]

            self.W = W = tf.reshape(tf.slice(W_target, tf.slice(W_starts, target_id, one), tf.slice(W_lengths, target_id, one)), [-1, 2*state_size])
            self.b = b = tf.slice(b_target, tf.slice(b_starts, target_id, one), tf.slice(b_lengths, target_id, one))

            self.logits = logits = tf.matmul(W, unbatched_states[i], False, True) + b
            self.l = l = tf.reshape(logits, [-1])
            exp_logits = tf.exp(l)
            summ = tf.reduce_sum(exp_logits)
            self.p_targets = p_targets = exp_logits / summ

            n_senses = tf.slice(n_senses_sorted_by_target_id_tf, target_id, [1])
            answer = tf.sparse_to_dense(sense_id, n_senses, 1.0, 0.0)

            p_target = tf.slice(p_targets, sense_id, one)
            p_target_safe = max(0.0001, p_target)
            loss += - tf.reduce_sum(tf.mul(answer, tf.log(p_target_safe)))
            # loss += - tf.log(p_target_safe)

            # accuracy
            n_correct += tf.cast(tf.equal(sense_id, tf.cast(tf.arg_max(l, 0), tf.int32)), tf.int32)

            if i == batch_size-1:
                tf.scalar_summary(['p_target'], p_target)
                tf.scalar_summary(['target_id'], tf.cast(target_id, tf.float32))
                tf.scalar_summary(['sense_id'], tf.cast(sense_id, tf.float32))
                tf.scalar_summary(['n_correct'], tf.cast(n_correct, tf.float32))
                tf.histogram_summary('logits', l)
                tf.histogram_summary('W_target', W_target)
                tf.histogram_summary('b_target', b_target)

        self.cost_op = tf.div(loss, batch_size)
        self.accuracy_op = tf.truediv(n_correct, batch_size)
        self.error_op = self.cost_op

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
        self.train_op = tf.train.AdagradOptimizer(0.2).minimize(self.error_op)
        self.summary_op = tf.merge_all_summaries()


def run_epoch(session, model, batch_size, train_data, val_data):
    train_cost = 0.
    train_acc = 0.
    val_cost = 0.
    val_acc = 0.

    summaries = []

    n_train_batches = 0
    train_batch_gen = batch_generator(batch_size, train_data, word_to_id['<pad>'], n_step_f, n_step_b)
    for batch in train_batch_gen:
        xf, xb, target_ids, sense_ids = batch

        batch_cost, batch_acc, summary, _ = session.run([model.cost_op, model.accuracy_op, model.summary_op, model.train_op], {
            model.train_inputs: xf,
            model.train_target_ids: target_ids,
            model.train_sense_ids: sense_ids
        })
        train_cost += batch_cost
        train_acc += batch_acc
        summaries.append(summary)
        n_train_batches += 1

    n_val_batches = 0
    val_batch_gen = batch_generator(batch_size, val_data, word_to_id['<pad>'], n_step_f, n_step_b)
    for batch in val_batch_gen:
        xf, xb, target_ids, sense_ids = batch
        batch_cost, batch_acc = session.run([model.cost_op, model.accuracy_op], {
            model.train_inputs: xf,
            model.train_target_ids: target_ids,
            model.train_sense_ids: sense_ids
        })
        val_cost += batch_cost
        val_acc += batch_acc
        n_val_batches += 1

    print 'Cost (train/val): %f/%f, Accuracy (train/val): %f/%f' \
          % (train_cost / n_train_batches, val_cost / n_val_batches, train_acc / n_train_batches, val_acc / n_val_batches )

    return summaries


if __name__ == '__main__':
    n_epochs = 500
    batch_size = 200
    train_data, val_data = train_test_split(train_ndata)

    init_emb = fill_with_gloves(word_to_id, 100)

    model = Model(batch_size, n_step_f, n_step_b, init_emb)

    session = tf.Session()
    session.run(tf.initialize_all_variables())

    summary_op = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter('/home/salomons/tmp/tf.log', session.graph_def, flush_secs=10)

    for i in range(n_epochs):
        print 'EPOCH: %d' % i
        summaries = run_epoch(session, model, batch_size, train_data, val_data)
        for batch_idx, summary in enumerate(summaries):
            writer.add_summary(summary, i*len(train_data)//batch_size + batch_idx)

