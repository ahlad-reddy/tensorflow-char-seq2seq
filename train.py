import tensorflow as tf
from helper import build_helper
from model import Model 
from hparams import hparams as hp

mode = 'train'
helper = build_helper(mode)
train_model = Model(helper=helper, mode=mode)

train_model.sess.run(helper.initializer)
epoch = 0
epoch_loss = 0
while epoch < hp.epochs:
    try:
        gs, input_seq, input_len, target_seq, target_len, outputs, loss, _ = train_model.sess.run([train_model.global_step, train_model.input_seq, train_model.input_len, train_model.target_seq, train_model.target_len, train_model.outputs, train_model.loss, train_model.train_op])
        print('Global Step: {}, Loss: {}'.format(gs, loss))
        print('Input: {}'.format(''.join(helper.ix_to_char[i] for i in input_seq[0][:input_len[0]+1])))
        print('Target: {}'.format(''.join(helper.ix_to_char[i] for i in target_seq[0][:target_len[0]+1])))
        print('Prediction: {}'.format(''.join(helper.ix_to_char[i] for i in outputs.sample_id[0][:target_len[0]+1])))
        print()
        epoch_loss += loss
    except tf.errors.OutOfRangeError:
        print(epoch_loss)
        epoch_loss = 0
        epoch += 1
        train_model.sess.run(helper.initializer)

        if epoch % 5 == 0:
            train_model.save_model()



