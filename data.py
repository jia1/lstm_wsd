import lxml.etree as et
import tensorflow as tf
import numpy as np
import collections
import re
import random
import nltk.stem.porter as porter
from nltk.stem.wordnet import WordNetLemmatizer
from itertools import groupby
import random


replace_target = re.compile("""<head.*?>.*</head>""")
replace_newline = re.compile("""\n""")
replace_dot = re.compile("\.")
replace_cite = re.compile("'")
rm_context_tag = re.compile('<.{0,1}context>')
rm_cit_tag = re.compile('\[[eb]quo\]')
rm_misc = re.compile("[\[\]\$`()%/]")

# stemmer = porter.PorterStemmer()


def clean_context(ctx):
    ctx = replace_target.sub(' <target> ', ctx)
    ctx = replace_newline.sub(' <eop> ', ctx)
    ctx = replace_dot.sub(' <eos> ', ctx)
    ctx = replace_cite.sub(' <cite> ', ctx)
    ctx = rm_cit_tag.sub(' ', ctx)
    ctx = rm_context_tag.sub('', ctx)
    ctx = rm_misc.sub('', ctx)
    return ctx


def split_context(ctx):
    word_list = re.split(', | +|\? |! |: |; ', ctx.lower())
    return word_list  #[stemmer.stem(word) for word in word_list]


def one_hot_encode(length, target):
    y = np.zeros(length, dtype=np.float32)
    y[target] = 1.
    return y


def load_senteval2_data(path, is_training):
    data = []
    parser = et.XMLParser(dtd_validation=True)
    doc = et.parse(path, parser)
    instances = doc.findall('.//instance')

    for instance in instances:
        answer = None
        context = None
        for child in instance:
            if child.tag == 'answer':
                senseid = child.get('senseid')
                if senseid == 'P' or senseid == 'U':  # ignore
                    pass
                else:
                    answer = senseid
            elif child.tag == 'context':
                context = et.tostring(child)
            else:
                raise ValueError('unknown child tag to instance')

        # if valid
        if (is_training and answer and context) or (not is_training and context):
            context = clean_context(context)
            x = {
                'id': instance.get('id'),
                'docsrc': instance.get('docsrc'),
                'context': context,
                'target_sense': answer,  # todo support multiple answers?
                'target_word': instance.get('id').split('.')[0],
            }
            data.append(x)

    return data


def get_lexelts(path):
    items = []
    parser = et.XMLParser(dtd_validation=True)
    doc = et.parse(path, parser)
    instances = doc.findall('.//lexelt')

    for instance in instances:
        items.append(instance.get('item'))

    return items

def target_to_lexelt_map(target_words, lexelts):
    assert len(target_words) == len(lexelts)

    res = {}
    for lexelt in lexelts:
        base = lexelt.split('.')[0]
        res[base] = lexelt

    return res


def build_sense_ids_for_all(data):
    counter = collections.Counter()
    for elem in data:
        counter.update([elem['answer']])

    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    senses, _ = list(zip(*count_pairs))
    sense_to_id = dict(zip(senses, range(len(senses))))

    return sense_to_id


def build_sense_ids(data):
    words = set()
    word_to_senses = {}
    for elem in data:
        target_word = elem['target_word']
        target_sense = elem['target_sense']

        if target_word not in words:
            words.add(target_word)
            word_to_senses.update({target_word: [target_sense]})

        else:
            if target_sense not in word_to_senses[target_word]:
                word_to_senses[target_word].append(target_sense)

    words = list(words)
    target_word_to_id = dict(zip(words, range(len(words))))
    target_sense_to_id = [dict(zip(word_to_senses[word], range(len(word_to_senses[word])))) for word in words]

    n_senses_from_word_id = dict([(target_word_to_id[word], len(word_to_senses[word])) for word in words])
    return target_word_to_id, target_sense_to_id, len(words), n_senses_from_word_id


def build_vocab(data):
    """
    :param data: list of dicts containing attribute 'context'
    :return: a dict with words as key and ids as value
    """
    counter = collections.Counter()
    for elem in data:
        counter.update(split_context(elem['context']))

    # remove infrequent words
    min_freq = 5
    filtered = [item for item in counter.items() if item[1]>=min_freq]

    count_pairs = sorted(filtered, key=lambda x: -x[1])
    words, _ = list(zip(*count_pairs))
    words += ('<pad>',)
    word_to_id = dict(zip(words, range(len(words))))

    return word_to_id


def convert_to_numeric(data, word_to_id, target_word_to_id, target_sense_to_id, n_senses_from_target_id):
    n_senses_sorted_by_target_id = [n_senses_from_target_id[target_id] for target_id in range(len(n_senses_from_target_id))]
    starts = (np.cumsum(np.append([0], n_senses_sorted_by_target_id)))[:-1]
    tot_n_senses = sum(n_senses_from_target_id.values())

    def get_tot_id(target_id, sense_id):
        return starts[target_id] + sense_id

    all_data = []
    target_tag_id = word_to_id['<target>']
    for instance in data:
        words = split_context(instance['context'])
        ctx_ints = [word_to_id[word] for word in words if word in word_to_id]
        stop_idx = ctx_ints.index(target_tag_id)
        xf = np.array(ctx_ints[:stop_idx])
        xb = np.array(ctx_ints[stop_idx+1:])[::-1]
        instance_id = instance['id']
        target_word = instance['target_word']
        target_sense = instance['target_sense']
        target_id = target_word_to_id[target_word]
        senses = target_sense_to_id[target_id]
        sense_id = senses[target_sense] if target_sense else -1

        instance = Instance()
        instance.id = instance_id
        instance.xf = xf
        instance.xb = xb
        instance.target_id = target_id
        instance.sense_id = sense_id
        instance.one_hot_labels = one_hot_encode(n_senses_from_target_id[target_id], sense_id)
        # instance.one_hot_labels = one_hot_encode(tot_n_senses, get_tot_id(target_id, sense_id))

        all_data.append(instance)

    return all_data

def group_by_target(ndata):
    res = {}
    for key, group in groupby(ndata, lambda inst: inst.target_id):
       res.update({key: list(group)})
    return res

def split_grouped(data, frac, min=None):
    assert frac > 0.
    assert frac < .5
    l = {}
    r = {}
    for target_id, instances in data.iteritems():
        # instances = [inst for inst in instances]
        random.shuffle(instances)   # optional
        n = len(instances)
        n_r = int(frac * n)
        if min and n_r < min:
            n_r = min
        n_l = n - n_r

        l[target_id] = instances[:n_l]
        r[target_id] = instances[-n_r:]

    return l, r


def batchify_grouped(gdata, n_step_f, n_step_b, pad_id, n_senses_from_target_id):
    res = {}
    for target_id, instances in gdata.iteritems():
        batch_size = len(instances)
        xfs = np.zeros([batch_size, n_step_f], dtype=np.int32)
        xbs = np.zeros([batch_size, n_step_b], dtype=np.int32)
        xfs.fill(pad_id)
        xbs.fill(pad_id)

        # x forward backward
        for j in range(batch_size):
            n_to_use_f = min(n_step_f, len(instances[j].xf))
            n_to_use_b = min(n_step_b, len(instances[j].xb))
            xfs[j, -n_to_use_f:] = instances[j].xf[-n_to_use_f:]
            xbs[j, -n_to_use_b:] = instances[j].xb[-n_to_use_b:]

        # labels
        labels = np.zeros([batch_size, n_senses_from_target_id[target_id]], np.float32)
        for j in range(batch_size):
            labels[j, instances[j].sense_id] = 1.

        res[target_id] = (xfs, xbs, labels)

    return res


class Instance:
    pass


def batch_generator(batch_size, data, pad_id, n_step_f, n_step_b):
    n_batches = len(data) // batch_size
    random.shuffle(data)

    for i in range(n_batches):
        batch = data[i * batch_size:(i+1) * batch_size]

        xfs = np.zeros([batch_size, n_step_f], dtype=np.int32)
        xbs = np.zeros([batch_size, n_step_b], dtype=np.int32)
        xfs.fill(pad_id)
        xbs.fill(pad_id)

        # x forward backward
        for j in range(batch_size):
            n_to_use_f = min(n_step_f, len(batch[j].xf))
            n_to_use_b = min(n_step_b, len(batch[j].xb))
            xfs[j, -n_to_use_f:] = batch[j].xf[-n_to_use_f:]
            xbs[j, -n_to_use_b:] = batch[j].xb[-n_to_use_b:]

        # labels
        target_ids = np.array([inst.target_id for inst in batch]).astype(np.int32)
        sense_ids = np.array([inst.sense_id for inst in batch]).astype(np.int32)
        # one_hot_labels = np.vstack([inst.one_hot_labels for inst in batch])

        # id
        instance_ids = [inst.id for inst in batch]


        yield (xfs, xbs, target_ids, sense_ids, instance_ids)


def write_submission_file(answers):
    pass



if __name__ == '__main__':
    # load data
    data = load_senteval2_data()

    # build vocab
    word_to_id = build_vocab(data)
    # target_word_to_id, target_sense_to_id = build_sense_ids(data)

    # make numeric
    # ndata = convert_to_numeric(data, word_to_id, target_word_to_id, target_sense_to_id)
    #
    # batch_generator(50, ndata, word_to_id['<pad>'])
