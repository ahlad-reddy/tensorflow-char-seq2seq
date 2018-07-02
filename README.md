# Tensorflow Character Sequence-to-Sequence

A Tensorflow Sequence-to-Sequence implementation for text generation. Training will learn character level associations from sentences within a dataset of txt files. Generation will recursively generate sentences from a starting sentence, using the previously generated sentence as the input. Requirements are Python 3, Tensorflow, and NLTK. Begin by installing the requirements.

    pip install -r requirements.txt

Then create a folder called data and include all your .txt files containing sentences. 

To train the model, execute

    python train.py

This will create checkpoints in your newly created models/ directory.

To generate text, edit the generate.py file. Point line 10 to your newly created checkpoint and line 13 to your first input text.

To generate, execute

    python generate.py

This will create a txt file called generated.txt with your newly generated text.