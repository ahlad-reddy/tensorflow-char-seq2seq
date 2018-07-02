import tensorflow as tf
import numpy as np
import os
from hparams import hparams as hp

class Model():
    def __init__(self, helper, mode='train'):
        self.helper = helper
        self.mode = mode
        self.vocab_size = len(self.helper.ix_to_char)
        if self.mode == 'train':
            self.keep_prob = hp.drop_rate
        else:
            self.keep_prob = 1
        self._build_graph()
        self._init_session()
        
    def _build_graph(self):
        self.global_step = tf.Variable(0, trainable=False)
        self._initialize_placeholders()
        self._model()
        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver(max_to_keep=3)

    def _initialize_placeholders(self):
        if self.mode=='train':
            self.input_seq = self.helper.input_seq
            self.input_len = self.helper.input_len
            self.target_seq = self.helper.target_seq
            self.target_len = self.helper.target_len
        else:
            self.input_seq = tf.placeholder(tf.int32, shape=(1, None))
            self.input_len = [tf.shape(self.input_seq, out_type=tf.int32)[1]]
        
    def _model(self):
        self.embedding = tf.get_variable("embedding", [self.vocab_size, hp.rnn_size])
        encoder_outputs, encoder_state= self._build_encoder()
        self.outputs, final_state= self._build_decoder(encoder_state)
        if self.mode == 'train':
            self._compute_loss()
        
    def _build_encoder(self):
        with tf.variable_scope("encoder") as encoder_scope:
            enc_emb_inp = tf.nn.embedding_lookup(self.embedding, self.input_seq)
            
            enc_cell = tf.contrib.rnn.LSTMCell(hp.rnn_size)
            enc_cell = tf.contrib.rnn.DropoutWrapper(enc_cell, output_keep_prob=self.keep_prob)
            enc_cell = tf.contrib.rnn.MultiRNNCell([enc_cell] * 2)

            encoder_outputs, encoder_state = tf.nn.dynamic_rnn(enc_cell, enc_emb_inp, dtype=tf.float32, sequence_length=self.input_len)
        return encoder_outputs, encoder_state
        
    def _build_decoder(self, encoder_state):
        with tf.variable_scope("decoder") as decoder_scope:
            dec_cell = tf.contrib.rnn.LSTMCell(hp.rnn_size)
            dec_cell = tf.contrib.rnn.DropoutWrapper(dec_cell, output_keep_prob=self.keep_prob)
            dec_cell = tf.contrib.rnn.MultiRNNCell([dec_cell] * 2)

            if self.mode == 'train':
                dec_emb_inp = tf.nn.embedding_lookup(self.embedding, self.target_seq)
                decoder_helper = tf.contrib.seq2seq.TrainingHelper(dec_emb_inp, self.target_len)
                max_iter = None
            elif self.mode == 'generate':
                start_tokens = tf.fill([tf.shape(self.input_seq)[0]], self.helper.sos_id)
                end_token = self.helper.eos_id
                decoder_helper = tf.contrib.seq2seq.SampleEmbeddingHelper(self.embedding, start_tokens, end_token)
                max_iter = 10 * tf.shape(self.input_seq)[1]

            decoder = tf.contrib.seq2seq.BasicDecoder(dec_cell, decoder_helper, encoder_state, output_layer=tf.layers.Dense(self.vocab_size))
            outputs, final_state, _ = tf.contrib.seq2seq.dynamic_decode(decoder, maximum_iterations=max_iter, scope=decoder_scope)
        return outputs, final_state

    def _compute_loss(self):
        self.logits = self.outputs.rnn_output
        target_weights = tf.sequence_mask(self.target_len, dtype=self.logits.dtype)
        self.loss = tf.contrib.seq2seq.sequence_loss(logits=self.logits, targets=self.target_seq[:, 1:], weights=target_weights)
        
        params = tf.trainable_variables()
        gradients = tf.gradients(self.loss, params)
        clipped_gradients, _ = tf.clip_by_global_norm(gradients, hp.max_grad_clip)
        optimizer = tf.train.AdamOptimizer(hp.lr)
        self.train_op = optimizer.apply_gradients(zip(clipped_gradients, params), global_step=self.global_step)

    def _init_session(self):
        self.sess = tf.Session()
        self.sess.run(self.init)
        
    def save_model(self, directory=None):
        save_path = self.saver.save(self.sess, os.path.join('models','model.ckpt'), global_step=self.global_step)
        return save_path

    def load_model(self, model_path):
        print('Loading model from %s' % model_path)
        self.saver.restore(self.sess, model_path)
