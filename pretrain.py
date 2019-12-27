import os
import json
import fire
import random
import pickle
import math
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
                     num_candidate_samples=-1,
                     weight=None):
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
    if weight is None:
        lin_w = tf.get_variable(name="lin_w", shape=[pre_logits_hidden_size, vocab_size],\
                                    dtype=tf.float32)
    else:
        lin_w = weight
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
                         batch_size, embed_size,
                         hidden_size, dropout,
                         num_candidate_samples,
                         maxlen, clip):
    """
        This creates language model tensorflow graph. It takes placeholder
        for input tokens, output_tokens (target), initial state for LSTM layers.
        Lanugage model graph has Embedding Layer followed by LSTM layers. Loss
        is calculated using sampled softmax layer of tensorflow.

        :params:
            - input_tokens: Placeholder for input tokens  [shape:(batch_size, None)]
            - output_tokens: Placeholder for output tokens (used as target)
                                [shape:(batch_size, None)]
            - initial_state: Initial states placeholder for feeding state in LSTM
                                Layers [shape:(num_layers, batch_size, hidden_size)]
            - num_layers: Number of LSTM Layers
            - max_vocab_size: Maximum Vocabulary size
            - vocab_freqs: Frequency of words
            - batch_size: Batch Size (should not be none)
            - embed_size: Embedding Dimensions
            - hidden_size: Hidden size of LSTM layers
            - dropout: Dropout to keep between Layers, same dropout is applied after
                        Embedding as well as between and after LSTM Layers
            - num_candidate_samples: Candidate Samples to consider for Sampled softmax
                            -1 to calculate complete softmax
            - maxlen: Sequence length of examples (bptt)
            - clip: clip gradients by `clip`

        :returns:
            - train_op: Training Op Tensorflow
            - training_flag: Var for training flag
            - sampled_loss: Sampled Loss Variable
            - loss: Complete Loss Variable
            - final_state: Output State of LSTMs
            - weights: Dictionay containing weights of Embedding and LSTM layers
            - learning_rate: Learning Rate Variable
    """
    bptt = tf.shape(input_tokens)[1]
    training_flag = tf.Variable(True)
    learning_rate = tf.Variable(20.0)
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
        rnn_output = tf.layers.dropout(
                                        rnn_output ,
                                        rate=dropout,
                                        training=training_flag,
                                        noise_shape=[batch_size, 1, hidden_size]
                                    )

        final_state_cs.append(final_state_c)
        final_state_hs.append(final_state_h)

    final_state_c = tf.stack(final_state_cs, 0)
    final_state_h = tf.stack(final_state_hs, 0)

    final_state = (final_state_c, final_state_h)
    # rnn_output = tf.layers.dropout(
    #                                 rnn_output ,
    #                                 rate=dropout,
    #                                 training=training_flag,
    #                                 noise_shape=[batch_size, 1, embed_size]
    #                             )

    weight = embedding_layer.weights[0]
    weight = tf.transpose(weight, [1, 0])
    # weight = None
    with tf.variable_scope("loss"):
        sampled_loss = _sampled_lm_loss(rnn_output, output_tokens,
                             max_vocab_size,
                             vocab_freqs=vocab_freqs,
                             num_candidate_samples=num_candidate_samples,
                             weight=weight)

    with tf.variable_scope("loss", reuse=True):
        loss = _sampled_lm_loss(rnn_output, output_tokens,
                             max_vocab_size,
                             vocab_freqs=vocab_freqs,
                             num_candidate_samples=-1,
                             weight=weight)

    with tf.variable_scope("optimizer"):
        # sampled_loss = loss
        t_vars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(sampled_loss*maxlen, t_vars),
                                                        clip)
        # train_op = tf.train.AdamOptimizer(learning_rate).apply_gradients(zip(grads, t_vars))
        train_op = tf.train.GradientDescentOptimizer(learning_rate).apply_gradients(zip(grads, t_vars))
        # train_op = tf.compat.v1.train.MomentumOptimizer(learning_rate, momentum=0.9).apply_gradients(zip(grads, t_vars))


    # Extract Weights
    weights = {}
    weights["embedding"] = embedding_layer.weights
    weights["lstm"] = [rnn_layer.weights for rnn_layer in rnn_layers]

    return train_op, training_flag, sampled_loss, loss,  final_state, weights,\
                learning_rate

def _run_epoch(X, y, epoch, session, sampled_loss, loss,
                num_layers,
                batch_size,
                hidden_size,
                input_placeholder,
                target_placeholder,
                initial_state_c,
                initial_state_h,
                learning_rate_var,
                learning_rate,
                train_op,
                final_state_c,
                final_state_h,
                training_flag,
                seq_length=45,
                train=False,
                lr_cosine_decay_params=None,
                print_progress=True):
    """
        Runs a single epoch of training or validation

        :params:
            - X: Input
            - y: Target
            - train: Training Flag (Dropouts are turned off by this)
            - print_progress: Print the progress (tqdm progress bar)
            - All other params are self expanatory or already described

        :outputs:
            - mean loss of all batches
    """
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
        # if lr_cosine_decay_params:
        #     learning_rate = lr_cosine_decay_params["learning_rate"]
        #     t_mul = lr_cosine_decay_params["t_mul"]
        #     steps = (epoch*max_steps + i + 1)
        #     cycles_completed = int(math.log(
        #                             steps * (t_mul - 1) /
        #                             lr_cosine_decay_params["first_decay_steps"] + 1
        #                             ) / math.log(t_mul)
        #                            )
        #     cycle = lr_cosine_decay_params["first_decay_steps"] * \
        #             ( t_mul**cycles_completed + lr_cosine_decay_params["first_decay_steps"]  )
        #     min_learning_rate = lr_cosine_decay_params["learning_rate"] * lr_cosine_decay_params["alpha"]
        #     learning_rate = min_learning_rate + 0.5 * (learning_rate - min_learning_rate) * \
        #                         math.cos( steps * math.pi / cycle )
            # learning_rate = tf.train.cosine_decay_restarts(
            #                 global_step=epoch*max_steps + i ,
            #                 **lr_cosine_decay_params)
        item = next(data_iterator)
        feed_dict = {input_placeholder: item[0],
                     target_placeholder: item[1],
                     initial_state_c: c,
                     initial_state_h: h,
                     training_flag:train,
                     learning_rate_var:learning_rate}
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


def pretrain_encoder(train_file, valid_file,\
                     save_folder='saved_model/base', tokenizer=None,
                     restore=False,
                     **kwargs):
    """
        Module for running the training and validation subroutines.

        :params:
            - train_file: Training File, File with sentences separated by newline
            - valid_file: Validation File, same format as above
            - save_folder: Folder to save output files and models
            - restore: Whether to restore from the save_folder (can be used
                        to finetune on a smaller dataset)
            - tokenizer: Tokenizer to use for tokenizing sentences into tokens
            - **kwargs: other params:
                        * batch_size
                        * hidden_size
                        * num_layers
                        * epochs
                        * seq_length
        :outputs:
            None
    """
    config = FW_CONFIG
    tokenizer = get_tokenizer(tokenizer) if tokenizer else None
    batch_size = kwargs.get("batch_size") or FW_CONFIG["batch_size"]
    hidden_size = kwargs.get("hidden_size") or FW_CONFIG["hidden_size"]
    num_layers = kwargs.get("num_layers") or FW_CONFIG["num_layers"]
    epochs = kwargs.get("epochs") or FW_CONFIG.pop("epochs")
    if "epochs" in FW_CONFIG:
        FW_CONFIG.pop("epochs")
    FW_CONFIG["num_candidate_samples"] = kwargs.get("num_candidate_samples") or FW_CONFIG["num_candidate_samples"]
    seq_length = FW_CONFIG.pop("seq_length")
    learning_rate = 1.0
    learning_rate_decay = 0.1
    lr_cosine_decay_params = {
            "learning_rate": learning_rate,
            "first_decay_steps": 2000,
            "t_mul": 2.0,
            "alpha": 0.01
    }
    tokenizer_json_file = os.path.join(save_folder, "tokenizer.json")
    # Load data and Batchify
    all_data = load_and_process_data(train_file, valid_file,
                                       max_vocab_size=config["max_vocab_size"],
                                       custom_tokenizer_function=tokenizer,
                                       tokenizer_json_file=tokenizer_json_file,
                                       restore_from=tokenizer_json_file if restore else None)

    word_freq, word_index, train_data, valid_data = all_data
    X_train, y_train = batchify(train_data, batch_size)
    X_valid, y_valid = batchify(valid_data, batch_size)

    # Save the Vocab and frequency files
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

    # Check max_vocab_size
    FW_CONFIG["max_vocab_size"] = min(len(word_index) + 1, FW_CONFIG["max_vocab_size"])
    print("Vocabulary Size: {}".format(FW_CONFIG["max_vocab_size"]))

    # Define Placeholder and Initial States
    inputs  = tf.placeholder(dtype=tf.int64, shape=(batch_size,None), name='input')
    targets = tf.placeholder(dtype=tf.int64, shape=(batch_size,None), name='target')
    initial_state_c  = tf.placeholder(dtype=tf.float32, shape=(num_layers, batch_size, hidden_size),\
                                    name='input_state_c')
    initial_state_h = tf.placeholder(dtype=tf.float32, shape=(num_layers, batch_size, hidden_size),\
                                    name='input_state_h')

    # Create the Graph
    train_op, training_flag, sampled_loss,\
    loss, rnn_states, weights, learning_rate_var = language_model_graph(inputs, targets,
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
                        "learning_rate_var":learning_rate_var,
                        "learning_rate":learning_rate,
                        "train_op": train_op,
                        "final_state_c": final_state_c,
                        "final_state_h": final_state_h,
                        "seq_length": seq_length,
                        "batch_size": batch_size,
                        "hidden_size":hidden_size,
                        "training_flag": training_flag,
                        "lr_cosine_decay_params": lr_cosine_decay_params}

    valid_losses = [1000]
    vars = tf.trainable_variables()
    vars = [i for i in vars if 'optimizer' not in i.name]
    saver = tf.train.Saver(vars)
    if restore:
        saver.restore(sess, os.path.join(save_folder, "model.ckpt"))
    for epoch in range(epochs):
        decay = (learning_rate_decay ** int((max(epoch - 5, 0)/2)))

        run_epoch_params['learning_rate'] = learning_rate * decay
        # Training Epoch
        train_loss = _run_epoch(X_train, y_train,
                                train=True,
                                epoch=epoch,
                                print_progress=True,
                                **run_epoch_params)
        # Valid Epoch
        valid_loss = _run_epoch(X_valid, y_valid,
                                train=False,
                                print_progress=False,
                                epoch=epoch,
                                **run_epoch_params)

        format_values = [epoch, train_loss, np.exp(train_loss),\
                                valid_loss, np.exp(valid_loss)]

        print("Epoch {0}, Train Loss {1:.2f}, Train Perplexity {2:.2f},\
                    Val Loss {3:.2f}, Val Perplexity {4:.2f}".format(*format_values))

        if valid_loss < min(valid_losses):
            saver.save(sess, os.path.join(save_folder, "model.ckpt"))
            numpy_weights = {}
            weights_ = weights
            for layer in weights:
                numpy_weights[layer] = sess.run(weights[layer])
            weights = weights_
            pickle.dump(numpy_weights, open(os.path.join(save_folder, "weights.pkl"), "wb"))

        valid_losses.append(valid_loss)


if __name__ == '__main__':
    fire.Fire(pretrain_encoder)
