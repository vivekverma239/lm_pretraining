import os
import json
import fire
import random
import pickle
from tqdm import tqdm
import tensorflow as tf
import numpy as np
from tensorflow.keras import layers

from data_utils import load_and_process_data, batchify, get_tokenizer, iterate
from config import FW_CONFIG

#python pretrain.py --train_file data/imdb/lm_train.txt --valid_file data/imdb/lm_val.txt

def _sampled_lm_loss(pre_logits, labels,
                     vocab_size,
                     vocab_freqs=None,
                     num_candidate_samples=-1):
    """
        Sampled Softmax loss to speedup training.
        Importance sampling is performed based on vocab_freqs

        :params:
            - pre_logits: Output of RNN Layer
            - labels: Target tokens
            - vocab_size: max vocab size
            - vocab_freqs: list of int containing frequency of
                            words (must of num_candidate_samples > 0)
            - num_candidate_samples (-1): Number of samples to sample for Softmax

        :output:
            - Tf variable loss
    """
    # Get the weight and biases
    pre_logits_hidden_size = pre_logits.get_shape()[-1]
    lin_w = tf.get_variable(name="lin_w", shape=[pre_logits_hidden_size, vocab_size],\
                                dtype=tf.float32)
    lin_b = tf.get_variable(name="lin_b", shape=[vocab_size],\
                                dtype=tf.float32)
    # Reshape Inputs and Lables
    inputs_reshaped = tf.reshape(pre_logits, [-1, int(pre_logits.get_shape()[2])])
    labels_reshaped = tf.reshape(labels, [-1])
    if num_candidate_samples > -1:
        # Sampled Softmax Case
        assert vocab_freqs is not None

        labels_reshaped = tf.expand_dims(labels_reshaped, -1)

        sampled = tf.nn.fixed_unigram_candidate_sampler(
                        true_classes=labels_reshaped,
                        num_true=1,
                        num_sampled=num_candidate_samples,
                        unique=True,
                        range_max=vocab_size,
                        unigrams=vocab_freqs)

        lm_loss = tf.nn.sampled_softmax_loss(
                                weights=tf.transpose(lin_w),
                                biases=lin_b,
                                labels=labels_reshaped,
                                inputs=inputs_reshaped,
                                num_sampled=num_candidate_samples,
                                num_classes=vocab_size,
                                sampled_values=sampled)

    else:
        # Normal Softmax Case
        logits = tf.nn.xw_plus_b(x=inputs_reshaped, weights=lin_w, biases=lin_b)
        lm_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(
                                    logits=logits, labels=labels_reshaped)

    lm_loss = tf.identity(tf.reduce_mean(lm_loss), name='lm_xentropy_loss')
    return lm_loss



def language_model_graph(input_tokens, output_tokens,
                         initial_state, num_layers,
                         max_vocab_size, vocab_freqs,
                         batch_size, embed_size,\
                         hidden_size, dropout, \
                         num_candidate_samples,
                         maxlen, clip):

    bptt = tf.shape(input_tokens)[1]
    training_flag = tf.Variable(True)
    embedding_layer = layers.Embedding(max_vocab_size, embed_size)
    rnn_layers = []

    for i in range(num_layers):
        rnn_layers.append(layers.CuDNNLSTM(units=hidden_size,
                                     return_sequences=True,
                                     return_state=True))

    embedded_input = embedding_layer(input_tokens)
    embedded_input = tf.layers.dropout(
                                    embedded_input ,
                                    rate=dropout,
                                    training=training_flag,
                                )

    states = []
    rnn_input = embedded_input
    input_state_cs =  initial_state[0]
    input_state_hs = initial_state[1]
    final_state_cs = []
    final_state_hs = []
    for i in range(num_layers):
        state_c, state_h = input_state_cs[i], input_state_hs[i]
        rnn_outputs = rnn_layers[i](rnn_input, initial_state=(state_c, state_h))
        rnn_output, final_state_c, final_state_h = rnn_outputs
        final_state_cs.append(final_state_c)
        final_state_hs.append(final_state_h)

    final_state_c = tf.stack(final_state_cs, 0)
    final_state_h = tf.stack(final_state_hs, 0)

    final_state = (final_state_c, final_state_h)
    rnn_output = tf.layers.dropout(
                                    rnn_output ,
                                    rate=dropout,
                                    training=training_flag,
                                )

    with tf.variable_scope("loss"):
        sampled_loss = _sampled_lm_loss(rnn_output, output_tokens,
                             max_vocab_size,
                             vocab_freqs=vocab_freqs,
                             num_candidate_samples=num_candidate_samples)

    with tf.variable_scope("loss", reuse=True):
        loss = _sampled_lm_loss(rnn_output, output_tokens,
                             max_vocab_size,
                             vocab_freqs=vocab_freqs,
                             num_candidate_samples=-1)

    t_vars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(sampled_loss*maxlen, t_vars),
                                                    clip)
    train_op = tf.train.AdamOptimizer().apply_gradients(grads)

    # Extract Weights
    weights = {}
    weights["embedding"] = embedding_layer.weights
    weights["lstm"] = [rnn_layer.weights for rnn_layer in rnn_layers]

    return train_op, training_flag, sampled_loss, loss,  final_state, weights

def _run_epoch(X, y, session, sampled_loss, loss,
                num_layers,
                batch_size,
                hidden_size,
                input_placeholder,
                target_placeholder,
                initial_state_c,
                initial_state_h,
                train_op,
                final_state_c,
                final_state_h,
                training_flag,
                seq_length=45,
                train=False,
                print_progress=True):

    data_iterator = iterate((X, y), seq_length=seq_length)
    computed_loss = []

    # Create a tqdm iterator
    max_steps = int(X.shape[1]/seq_length)
    if print_progress:
        tqdm_range = tqdm(list(range(max_steps)))
    else:
        tqdm_range = range(max_steps)

    batch_size = X.shape[0]
    # Intial LSTM State
    c = np.zeros((num_layers, batch_size, hidden_size), dtype=np.float32)
    h = np.zeros((num_layers, batch_size, hidden_size), dtype=np.float32)
    # Iterate over data
    for i in tqdm_range:
        item = next(data_iterator)
        feed_dict = {input_placeholder: item[0],
                     target_placeholder: item[1],
                     initial_state_c: c,
                     initial_state_h: h,
                     training_flag:train}
        if train:
            ops = [train_op, sampled_loss, final_state_c, final_state_h]
            _, loss_, c, h = session.run(ops, feed_dict=feed_dict)
        else:
            ops = [loss, final_state_c, final_state_h]
            loss_, c, h = session.run(ops, feed_dict=feed_dict)

        computed_loss.append(loss_)
        if print_progress:
            tqdm_range.set_description('Loss {}'.format(str(round(np.mean(computed_loss),2))))

    return np.mean(computed_loss)


def pretrain_encoder(train_file, valid_file, test_file=None, config=FW_CONFIG,\
                     save_folder='saved_model/base', tokenizer=None):

    tokenizer = get_tokenizer(tokenizer) if tokenizer else None
    batch_size = FW_CONFIG["batch_size"]
    hidden_size = FW_CONFIG["hidden_size"]
    num_layers = FW_CONFIG["num_layers"]
    epochs = FW_CONFIG.pop("epochs")
    seq_length = FW_CONFIG.pop("seq_length")
    # Load data and Batchify
    all_data = load_and_process_data(train_file, valid_file,
                                       test_file,
                                       max_vocab_size=config["max_vocab_size"],
                                       custom_tokenizer_function=tokenizer)
    if test_file:
        word_freq, word_index, train_data, valid_data, test_data = all_data
        X_train, y_train = batchify(train_data, batch_size)
        X_valid, y_valid = batchify(valid_data, batch_size)
        X_test, y_test = batchify(test_data, batch_size)
    else:
        word_freq, word_index, train_data, valid_data = all_data
        X_train, y_train = batchify(train_data, batch_size)
        X_valid, y_valid = batchify(valid_data, batch_size)

    FW_CONFIG["max_vocab_size"] = min(len(word_index) + 1, FW_CONFIG["max_vocab_size"])
    # Define Placeholder and Initial States
    inputs  = tf.placeholder(dtype=tf.int64, shape=(batch_size,None), name='input')
    targets = tf.placeholder(dtype=tf.int64, shape=(batch_size,None), name='target')
    initial_state_c  = tf.placeholder(dtype=tf.float32, shape=(num_layers, batch_size, hidden_size),\
                                    name='input_state_c')
    initial_state_h = tf.placeholder(dtype=tf.float32, shape=(num_layers, batch_size, hidden_size),\
                                    name='input_state_h')

    # Create the Graph
    train_op, training_flag, sampled_loss, loss, rnn_states, weights = language_model_graph(inputs, targets,
                                                     (initial_state_c, initial_state_h),
                                                     vocab_freqs=word_freq,
                                                      **config)

    final_state_c, final_state_h = rnn_states
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    # Define run epoch function params (passed as kwargs)
    run_epoch_params = {"session": sess,
                        "sampled_loss": sampled_loss,
                        "loss": loss,
                        "num_layers": num_layers,
                        "input_placeholder": inputs,
                        "target_placeholder": targets,
                        "initial_state_c": initial_state_c,
                        "initial_state_h": initial_state_h,
                        "train_op": train_op,
                        "final_state_c": final_state_c,
                        "final_state_h": final_state_h,
                        "seq_length": seq_length,
                        "batch_size": batch_size,
                        "hidden_size":hidden_size,
                        "training_flag": training_flag}

    valid_losses = []
    for epoch in range(epochs):
        # Training Epoch
        train_loss = _run_epoch(X_train, y_train,
                                train=True,
                                print_progress=True,
                                **run_epoch_params)
        # Valid Epoch
        valid_loss = _run_epoch(X_valid, y_valid,
                                train=False,
                                print_progress=False,
                                **run_epoch_params)

        format_values = [epoch, train_loss, np.exp(train_loss),\
                                valid_loss, np.exp(valid_loss)]

        print("Epoch {0}, Train Loss {1:.2f}, Train Perplexity {2:.2f},\
                    Val Loss {3:.2f}, Val Perplexity {4:.2f}".format(*format_values))

        valid_losses.append(valid_losses)

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    # Saving config and word_index file
    json.dump(word_index, open(os.path.join(save_folder, "word_index.json"), "w"))
    json.dump(word_freq, open(os.path.join(save_folder, "word_freq.json"), "w"))

    json.dump(FW_CONFIG, open(os.path.join(save_folder, "config.json"), "w"))

    # Arranging tokens in alist, this will go in vocab file
    vocab = [" "] + [i[0] for i in sorted(word_index.items(), key=lambda x: x[1])][:FW_CONFIG["max_vocab_size"]+1]
    open(os.path.join(save_folder, "vocab.txt"), "w").write("\n".join(vocab))

    open(os.path.join(save_folder, "word_freqs.txt"), "w").write("\n".join(word_index))
    numpy_weights = weights
    for layer in numpy_weights:
        numpy_weights[layer] = sess.run(weights[layer])
    pickle.dump(numpy_weights, open(os.path.join(save_folder, "weights.pkl"), "wb"))

if __name__ == '__main__':
    fire.Fire(pretrain_encoder)
