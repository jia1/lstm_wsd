from data import *
from glove import *
import tensorflow as tf
import numpy as np
import unittest

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


class DataTest(unittest.TestCase):
    def setUp(self):
        self.n_targets = len(np.unique([inst.target_id for inst in train_ndata]))

    def test_data_not_empty(self):
        self.assertTrue(len(train_data) > 0, 'Train data empty')
        self.assertTrue((len(train_ndata) > 0, 'numeric train data empty'))

    def test_group_data_by_target(self):
        grouped = group_by_target(train_ndata)
        self.assertEqual(self.n_targets, len(grouped))

    def test_split_grouped(self):
        grouped = group_by_target(train_ndata)
        tr, te = split_grouped(grouped, 0.2, 2)
        for target_id, vals in grouped.iteritems():
            n_g = len(vals)
            n_split = len(tr[target_id]) + len(te[target_id])
            self.assertEqual(n_g, n_split)


class FeatureTest(unittest.TestCase):
    def test_variable_scope(self):
        with tf.Session() as se:
            initer = tf.constant_initializer(2.)

            x = tf.placeholder(tf.float32, [None])

            def setup_ops(scope_name, size):
                with tf.variable_scope(scope_name, initializer=initer):
                    w = tf.get_variable('w', [size])

                y = tf.mul(w, x)
                return y

            y1_op = setup_ops('one', 2)
            y2_op = setup_ops('two', 3)

            x1 = np.array([1., 2.])
            x2 = np.array([1., 2., 3.])

            tf.initialize_all_variables().run(session=se)

            y1 = se.run([y1_op], {x: x1}, )
            y2 = se.run([y2_op], {x: x2})

            self.assertEqual(np.sum(y1), 6)
            self.assertEqual(np.sum(y2), 12)

    def test_diff_input_len(self):
        with tf.Session() as se:
            def setup_ops(len, reuse):
                with tf.variable_scope('y', reuse, tf.constant_initializer()):
                    y = tf.get_variable('y', [1])
                    if reuse:
                        y = tf.assign(y, [2.])
                for i in range(len):
                    y = y + 1
                return y

            # print tf.get_variable_scope().reuse
            y1_op = setup_ops(5, None)
            y2_op = setup_ops(8, True)
            y1_post_op = setup_ops(7, True)


            tf.initialize_all_variables().run(session=se)

            y1 = se.run([y1_op])
            y2 = se.run([y2_op])
            y1_post = se.run([y1_post_op])

            self.assertEqual(y1[0], 5.)
            self.assertEqual(y2[0], 10)
            self.assertEqual(y1_post[0], 9.)

    def test_None_len(self):
        pass


class ModelTest(unittest.TestCase):
    def setUp(self):
        self.data = group_by_target(train_ndata)
        self.se = tf.Session()


if __name__ == '__main__':
    unittest.main()
