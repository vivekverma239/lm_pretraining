# Language Model Pretraining for NLP Tasks
This repo is to see the effect of Language Model Pretraining on common NLP tasks. The idea is also to provide
a simple interface to use pretrained lanugage models in keras.

## General Idea

The general idea is pretty simple, pretrain LSTM layers on language modelling task then use the trained weights in downstream NLP task. It is just same as ULMFIT but can be used as a model in keras.

## How to use?
There are two steps to it, first pretrain the LSTM encoder and then simply use it Keras Model.
#### Pretrain LSTM Encoder:
```bash
python pretrain.py --train_file TRAIN_FILE --valid_file VAL_FILE --tokenizer [TOKENIZER]
```

**Params:**
  - **TRAIN_FILE:** Training file for Language Model, each sentence should be in a seperate line.
  - **VAL_FILE:** Validation file for Language Model, same format as above
  - **TOKENIZER [nltk]:** (spacy/nltk) Which tokenizer to use, nltk is by default used
  - **config params:**
    - batch_size: Batch size for training and evaluation (default 32)
    - hidden_size: LSTM hidden size (default 500)
    - num_layers: Number of LSTM Layers (default 1)
    - epochs: Number of epochs to train (default 10)
    - seq_length: BPTT for LM training (default: 70)
    - max_vocab_size: Max vocab size for embedding (default: 60000)
    - embed_size: Embedding Size (default: 500)
    - dropout : Dropout (default: 0.5)
    - num_candidate_samples: Number of candidates for sampled Softmax (default: 2048)
    - clip: Clip Gradient to (default: 0.25)

#### Using the pretrained LSTM Encoder:
```python
from keras import layers
import keras.backend as K
from keras.callbacks import Callback
# Model expects a sequence of tokens (words) as input
input_ = layers.Input(shape=(maxlen,), dtype=tf.string)
# pretrained_model_path -> Path where pretrained model is saved
pretrained_model = PretrainedLSTM(pretrained_model_path, input_, return_sequences=False)
encoder_output = pretrained_model.outputs[0]
final_output = layers.Dense(1, activation="sigmoid")(encoder_output)

model = Model(inputs=[input_], outputs=[final_output])
model.compile("adam", loss="binary_crossentropy" , metrics=['acc'])

# This is needed to initialize word to idx lookup layer
class TableInitializerCallback(Callback):
    """ Initialize Tables """
    def on_train_begin(self, logs=None):
        K.get_session().run(tf.tables_initializer())
callbacks = [TableInitializerCallback()]
# Finally fit
model.fit(x_train, y_train, epochs=10, callbacks=callbacks)
```

## Examples:
- IMDB Movie Review Dataset
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1VHZvEExiwEiFlO6fa1gAasowPSv9kSlP)
