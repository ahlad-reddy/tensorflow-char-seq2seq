import tensorflow as tf
import numpy as np
from helper import build_helper
from model import Model 


mode = 'generate'
helper = build_helper(mode)
generate_model = Model(helper=helper, mode='generate')
generate_model.load_model('models/model.ckpt-260')
eos_id = helper.char_to_ix['>']

input_seq = 'On ends of good and evil.>'
input_seq = [helper.char_to_ix[c] for c in input_seq]
generated = open('generated.txt', 'w')

for i in range(1000):
    if input_seq[-1] != eos_id: np.append(input_seq, eos_id)
    
    outputs = generate_model.sess.run(generate_model.outputs, feed_dict={ generate_model.input_seq: [input_seq] })
    output_seq = outputs.sample_id[0]
    text = ''.join(helper.ix_to_char[i] for i in output_seq).strip('>')
    print(i)
    print(text)
    generated.write(text)
    generated.write('\n')

    input_seq = output_seq
generated.close()

