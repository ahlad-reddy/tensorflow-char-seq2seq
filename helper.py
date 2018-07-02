import tensorflow as tf
import numpy as np
import glob
import nltk
from collections import namedtuple
from hparams import hparams as hp


class Helper(namedtuple("Helper", ("initializer", "input_seq", "input_len", "target_seq", "target_len", "ix_to_char", "char_to_ix", "sos_id", "eos_id"))):
    pass


def build_helper(mode):
    files = glob.glob('data/*.txt')
    print('Loaded {} files'.format(len(files)))

    vocab = set(['<', '>'])
    for f in files:
        data = open(f, 'r').read()
        vocab = vocab.union(data)
    vocab = sorted(vocab)
    ix_to_char = { i: ch for i, ch in enumerate(vocab)}
    char_to_ix = { ch: i for i, ch in ix_to_char.items() }
    sos_id = char_to_ix['<']
    eos_id = char_to_ix['>']
    print('Number of Characters: {}'.format(len(vocab)))

    if mode == 'train':
        data_pairs = []
        for f in files:
            data = open(f, 'r').read()
            sentences = nltk.sent_tokenize(data)
            data_pairs += [(sentences[i]+'>', '<'+sentences[i+1]+'>') for i in range(len(sentences)-1)]
        print('Created {} pairs'.format(len(data_pairs)))

        def generator():
            for p in data_pairs:
                d = ((np.array(list(map(char_to_ix.get, p[0]))), len(p[0])-1), (np.array(list(map(char_to_ix.get, p[1]))), len(p[1])-1))
                yield d

        ds = tf.data.Dataset.from_generator(generator, ((tf.int32, tf.int32), (tf.int32, tf.int32)))
        ds = ds.shuffle(1000)
        ds = ds.padded_batch(hp.batch_size, padded_shapes=((tf.TensorShape([None]), tf.TensorShape([])), (tf.TensorShape([None]), tf.TensorShape([]))))
        ds = ds.prefetch(1)

        iterator = ds.make_initializable_iterator()
        initializer = iterator.initializer
        ((input_seq, input_len), (target_seq, target_len)) = iterator.get_next()
    elif mode == 'generate':
        initializer = None
        input_seq = None
        input_len = None
        target_seq = None
        target_len = None
    
    return Helper(initializer=initializer, input_seq=input_seq, input_len=input_len, target_seq=target_seq, target_len=target_len, ix_to_char=ix_to_char, char_to_ix=char_to_ix, sos_id=sos_id, eos_id=eos_id)
        
