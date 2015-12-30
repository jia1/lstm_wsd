from data import *
from glove import *
import tensorflow as tf
import tensorflow.models.rnn.rnn_cell as rnn_cell
from sklearn.cross_validation import train_test_split


class Model:
    def __init__(self, is_training, conf, n_senses_from_target_id, word_to_id, init_word_vecs=None, skip_train_emb=0):
        batch_size = conf['batch_size']
        n_step_f = conf['n_step_f']
        n_step_b = conf['n_step_b']

        n_units = conf['n_lstm_units']
        n_layers = conf['n_layers']
        forget_bias = conf['forget_bias']
        train_init_state = conf['train_init_state']

        emb_base_std = conf['emb_base_std']
        input_keep_prob = conf['input_keep_prob']
        keep_prob = conf['keep_prob']
        embedding_size = conf['embedding_size']

        state_size = conf['state_size']

        lr_start = 2.0
        lr_decay_factor = 0.96
        lr_min = 0.05

        print 'n_step forward/backward: %d / %d' % (n_step_f, n_step_b)

        self.dbg = {}
        self.batch_size = batch_size
        self.is_training = is_training

        self.inputs_f = tf.placeholder(tf.int32, shape=[batch_size, n_step_f])
        self.inputs_b = tf.placeholder(tf.int32, shape=[batch_size, n_step_b])
        self.train_target_ids = train_target_ids = tf.placeholder(tf.int32, shape=[batch_size])
        self.train_sense_ids = train_sense_ids = tf.placeholder(tf.int32, shape=[batch_size])

        global_step = tf.Variable(1.0, trainable=False)

        tot_n_senses = sum(n_senses_from_target_id.values())
        # self.train_labels = labels = tf.placeholder(tf.float32, shape=[batch_size, tot_n_senses])

        vocab_size = len(word_to_id)

        def embedding_initializer(vec, dtype):
            return init_word_vecs if init_word_vecs is not None else tf.random_uniform([vocab_size, embedding_size],
                                                                                       -.1, .1, dtype)

        with tf.variable_scope('emb'):
            self.dbg['embeddings'] = embeddings = tf.get_variable('embeddings', [vocab_size, embedding_size],
                                                                  initializer=embedding_initializer,
                                                                  trainable=conf['train_embeddings'])

        mean_embeddings = tf.reduce_mean(embeddings, 0, keep_dims=True)
        self.dbg['std_emb'] = std_embeddings = tf.sqrt(tf.reduce_mean(tf.square(embeddings - mean_embeddings), 0))

        print 'Avg n senses per target word: ' + str(tot_n_senses / len(n_senses_from_target_id))

        n_senses_sorted_by_target_id = [n_senses_from_target_id[target_id] for target_id
                                        in range(len(n_senses_from_target_id))]
        n_senses_sorted_by_target_id_tf = tf.constant(n_senses_sorted_by_target_id, tf.int32)
        _W_starts = (np.cumsum(np.append([0], n_senses_sorted_by_target_id)) * state_size)[:-1]
        _W_lenghts = np.array(n_senses_sorted_by_target_id) * state_size
        W_starts = tf.constant(_W_starts, tf.int32)
        W_lengths = tf.constant(_W_lenghts, tf.int32)

        _b_starts = (np.cumsum(np.append([0], n_senses_sorted_by_target_id)))[:-1]
        _b_lengths = np.array(n_senses_sorted_by_target_id)
        b_starts = tf.constant(_b_starts, tf.int32)
        b_lengths = tf.constant(_b_lengths, tf.int32)

        with tf.variable_scope('target_params', initializer=tf.random_uniform_initializer(-.1, .1)):
            W_target = tf.get_variable('W_target', [tot_n_senses * state_size], dtype=tf.float32)
            b_target = tf.get_variable('b_target', [tot_n_senses], dtype=tf.float32,
                                       initializer=tf.constant_initializer(0.0))

        with tf.variable_scope("forward"):
            f_lstm = rnn_cell.BasicLSTMCell(n_units,
                                            forget_bias=forget_bias)  # LSTMCell(n_units, embedding_size, use_peepholes=True, initializer=tf.random_uniform_initializer(-.1, .1))
            if is_training:
                f_lstm = rnn_cell.DropoutWrapper(f_lstm, input_keep_prob=input_keep_prob)
            f_lstm = rnn_cell.MultiRNNCell([f_lstm] * n_layers)

            f_state = tf.get_variable('f_init_state', [batch_size, 2 * n_units * n_layers]) \
                if train_init_state else f_lstm.zero_state(batch_size, tf.float32)

            inputs_f = tf.split(1, n_step_f, self.inputs_f)
            for time_step, inputs_ in enumerate(inputs_f):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                emb = tf.nn.embedding_lookup(embeddings, tf.squeeze(inputs_))
                if is_training:
                    emb = emb + std_embeddings * tf.random_normal([batch_size, embedding_size], stddev=emb_base_std)
                _, f_state = f_lstm(emb, f_state)

        with tf.variable_scope("backward"):
            b_lstm = rnn_cell.BasicLSTMCell(n_units,
                                            forget_bias=forget_bias)  # LSTMCell(n_units, embedding_size, use_peepholes=True, initializer=tf.random_uniform_initializer(-.1, .1))
            if is_training:
                b_lstm = rnn_cell.DropoutWrapper(b_lstm, input_keep_prob=input_keep_prob)
            b_lstm = rnn_cell.MultiRNNCell([b_lstm] * n_layers)

            b_state = tf.get_variable('b_init_state', [batch_size, 2 * n_units * n_layers]) \
                if train_init_state else  b_lstm.zero_state(batch_size, tf.float32)

            inputs_b = tf.split(1, n_step_b, self.inputs_b)
            for time_step, inputs_ in enumerate(inputs_b):
                if time_step > 0:
                    tf.get_variable_scope().reuse_variables()
                emb = tf.nn.embedding_lookup(embeddings, tf.squeeze(inputs_))
                if is_training:
                    emb = emb + std_embeddings * tf.random_normal([batch_size, embedding_size],
                                                                  stddev=emb_base_std)  # tf.nn.dropout(emb, emb_keep_prop)
                _, b_state = b_lstm(emb, b_state)

        f_state = tf.slice(tf.split(1, n_layers, f_state)[-1], [0, n_units], [batch_size, n_units])
        b_state = tf.slice(tf.split(1, n_layers, b_state)[-1], [0, n_units], [batch_size, n_units])

        state = tf.concat(1, [f_state, b_state])
        if is_training:
            state = tf.nn.dropout(state, keep_prob)

        # hidden layer
        with tf.variable_scope('hidden'):
            hidden = rnn_cell.linear(state, state_size, True)
            if is_training:
                hidden = tf.nn.dropout(hidden, keep_prob)
        y_hidden = tf.nn.tanh(hidden)

        loss = tf.Variable(0., trainable=False)
        n_correct = tf.Variable(0, trainable=False)

        unbatched_states = tf.split(0, batch_size, y_hidden)
        unbatched_target_ids = tf.split(0, batch_size, train_target_ids)
        unbatched_sense_ids = tf.split(0, batch_size, train_sense_ids)
        one = tf.constant(1, tf.int32, [1])

        self.predictions = tf.Variable(tf.zeros([batch_size], dtype=tf.int64), trainable=False)

        # make predictions for all instances in batch
        for i in range(batch_size):
            target_id = unbatched_target_ids[i]  # tf.split(train_target_ids, i, [1])
            sense_id = unbatched_sense_ids[i]

            self.dbg['W'] = W = tf.reshape(
                tf.slice(W_target, tf.slice(W_starts, target_id, one), tf.slice(W_lengths, target_id, one)),
                [-1, state_size])
            self.dbg['b'] = b = tf.slice(b_target, tf.slice(b_starts, target_id, one),
                                         tf.slice(b_lengths, target_id, one))

            # self.dbg['ub_states'] = unbatched_states[i]
            # self.dbg['ub_states.shape'] = tf.shape(unbatched_states[i])

            # self.dbg['pre_b'] = tf.squeeze(tf.matmul(W, unbatched_states[i], False, True))
            self.dbg['logits'] = logits = tf.squeeze(tf.matmul(W, unbatched_states[i], False, True)) + b

            predicted_sense = tf.arg_max(logits, 0, name='prediction')
            self.predictions = tf.scatter_update(self.predictions, tf.constant(i, dtype=tf.int64), predicted_sense)

            self.dbg['exp_logits'] = exp_logits = tf.exp(logits)
            summ = tf.reduce_sum(exp_logits)
            self.dbg['p_targets'] = p_targets = exp_logits / summ

            n_senses = tf.slice(n_senses_sorted_by_target_id_tf, target_id, [1])
            answer = tf.sparse_to_dense(sense_id, n_senses, 1.0, 0.0)

            p_target = tf.slice(p_targets, sense_id, one)
            # p_target_safe = max(0.0001, p_target)
            self.dbg['p_targets_safe'] = p_targets_safe = max(0.0001, p_targets)
            self.dbg['mul'] = mul = tf.mul(answer, tf.log(p_targets_safe))
            loss += - tf.reduce_sum(mul)
            # loss += - tf.log(p_target_safe)

            # accuracy
            n_correct += tf.cast(tf.equal(sense_id, tf.cast(tf.arg_max(logits, 0), tf.int32)), tf.int32)

            # if i == batch_size-1:
            #     tf.scalar_summary(['p_target'], p_target)
            #     tf.scalar_summary(['n_correct'], tf.cast(n_correct, tf.float32))
            #     tf.histogram_summary('logits', logits)
            #     tf.histogram_summary('W_target', W_target)
            #     tf.histogram_summary('b_target', b_target)
        self.dbg['predictions'] = self.predictions

        self.cost_op = tf.div(loss, batch_size)
        self.accuracy_op = tf.div(tf.cast(n_correct, tf.float32), batch_size)
        self.error_op = self.cost_op

        if not is_training:
            return

        # grads = tf.gradients(self.cost_op, W_target)
        # tf.histogram_summary('grad_W_target', grads[0])
        # tf.scalar_summary('frac_0_grad_W', tf.nn.zero_fraction(grads[0]))

        print 'TRAINABLE VARIABLES'
        tvars = tf.trainable_variables()
        for tvar in tvars:
            print tvar.name

        max_grad_norm = 10
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost_op, tvars), max_grad_norm)
        self.lr = tf.maximum(lr_min, tf.train.exponential_decay(lr_start, global_step, 60, lr_decay_factor))
        optimizer = tf.train.MomentumOptimizer(self.lr, 0.1)

        # scaling down the learning for the embedings in the beginning
        # w = tf.constant(should_update, shape=[vocab_size, embedding_size])
        # w_embedding = tf.select(w, tf.zeros([vocab_size, embedding_size]), tf.ones([vocab_size, embedding_size]))
        # if conf['train_embeddings']:
        #     self.dbg['should_update'] = should_update = tf.to_float(tf.less(tf.to_int32(global_step), skip_train_emb))
        #     for i, tvar in enumerate(tvars):
        #         if tvar.name == 'model/emb/embeddings:0':
        #             grads[i] = tf.mul(grads[i], should_update)
        # self.dbg['grad_embeddings'] = tf.convert_to_tensor(grads[i])

        self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=global_step)
        self.train_op_no_emb = optimizer.apply_gradients(zip(grads[1:], tvars[1:]), global_step=global_step)

        self.summary_op = tf.merge_all_summaries()


def debug(model, session, feed_dict):
    for name, op in model.dbg.iteritems():
        value = session.run(op, feed_dict)
        print '::: %s :::: \n%s' % (name, np.array_str(value))


def debug_op(op, session, feed_dict):
    value = session.run(op, feed_dict)
    print value


def run_epoch(session, model, conf, data_, mode, word_to_id, freeze_emb=False):
    if mode == 'train':
        ops = [model.cost_op, model.accuracy_op, model.lr]
        if freeze_emb:
            print 'Embeddings frozen'
            ops += [model.train_op_no_emb]
        else:
            ops += [model.train_op]
    elif mode == 'val':
        ops = [model.cost_op, model.accuracy_op]
    else:
        raise ValueError('unknown mode')

    cost = 0.
    accuracy = 0.
    lr = 0.0
    summaries = []

    n_batches = 0
    for batch in batch_generator(mode == 'train', conf['batch_size'], data_, word_to_id['<pad>'], conf['n_step_f'],
                                 conf['n_step_b'], permute_order=conf.get('permute_input_order'),
                                 word_drop_rate=conf.get('word_drop_rate')):
        xf, xb, target_ids, sense_ids, instance_ids = batch
        feeds = {
            model.inputs_f: xf,
            model.inputs_b: xb,
            model.train_target_ids: target_ids,
            model.train_sense_ids: sense_ids
        }

        # debug(model, session, feeds)
        # debug_op(model.dbg['grad_embeddings'], session, feeds)

        fetches = session.run(ops, feeds)

        cost += fetches[0]
        accuracy += fetches[1]
        if mode == 'train':
            lr += fetches[2]

        n_batches += 1

    cost_epoch = cost / n_batches
    accuracy_epoch = accuracy / n_batches
    lr /= n_batches
    print '%s --> \tcost: %f, \taccuracy: %f, \tlr: %f' % (mode.upper(), cost_epoch, accuracy_epoch, lr)

    # if mode == 'train':
    #     return summaries

    return cost_epoch, accuracy_epoch

