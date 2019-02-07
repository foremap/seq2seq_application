#!/usr/bin/env python3
import os
import matplotlib.pylab as plt

from models import simple_s2s, lstm_s2s

from seq2seq.models import SimpleSeq2Seq

from keras.models import Sequential
from keras.layers import TimeDistributed, Masking
from keras.layers.embeddings import Embedding
from keras.callbacks import Callback
from keras.utils.vis_utils import plot_model
import keras.backend as K

import numpy as np
from preprocessing import preprocess

current_dir = os.path.dirname(os.path.realpath(__file__))

def get_loss(mask_value):
    mask_value = K.variable(mask_value)
    def masked_categorical_crossentropy(y_true, y_pred):
        # find out which timesteps in `y_true` are not the padding character 'PAD'
        mask = K.all(K.equal(y_true, mask_value), axis=-1)
        mask = 1 - K.cast(mask, K.floatx())
        # multiply categorical_crossentropy with the mask
        loss = K.sparse_categorical_crossentropy(y_true, y_pred) * mask
        # take average w.r.t. the number of unmasked entries
        return K.sum(loss) / K.sum(mask)
    return masked_categorical_crossentropy

class NBatchLogger(Callback): 
    """ A Logger that log average performance per `display` steps. """ 
    def __init__(self, display, data_handler): 
        self.step = 0 
        self.display = display
        self.data_handler = data_handler

    def on_batch_end(self, batch, logs={}): 
        self.step += 1 
        if self.step % self.display == 0: 
            tr_x, tr_y = self.data_handler.gen_all()
            tr_x = tr_x[:10]
            tr_y = tr_y[:10]
            res_y = self.model.predict(tr_x)
            print('\n******************************')
            for i in range(10):
                print('src: ', self.data_handler.seq2text(tr_x[i], is_target=False, is_logits=False))
                print('output: ', self.data_handler.seq2text(res_y[i], is_target=True, is_logits=True))
                print('target: ', self.data_handler.seq2text(tr_y[i], is_target=True, is_logits=False))
                print('---------------------')


def sample():
    config = {
        "input_length": 8,
        "input_dim": 1,
        "output_length": 8,
        "output_dim": 1,
        # model params
        "hidden_dim": 20,
        "depth": 5
    }

    a = np.random.random(1000)
    x = np.array([np.sin([[p] for p in np.arange(0, 0.8, 0.1)] + aa) for aa in a])
    y = -x
    print(x.shape)
    # training
    model = simple_s2s(**config)
    model.fit(x, y, epochs=10, batch_size=32)

    # testing
    x_test = np.array([np.sin([[p] for p in np.arange(0, 0.8, 0.1)] + aa) for aa in np.arange(0, 1.0, 0.1)])
    y_test = -x_test
    print(model.evaluate(x_test, y_test, batch_size=32))

    # predict
    predicted = model.predict(x_test, batch_size=32)

    plt.plot(np.arange(0, 0.8, 0.1), [xx[0] for xx in x_test[9]])
    plt.plot(np.arange(0, 0.8, 0.1), [xx[0] for xx in predicted[9]])
    plt.show()


# 對聯應用
def couplet():
    maxlen = 30 #length of input sequence and output sequence
    max_features = 10000 #number of words
    embedding_dim = 256 #word embedding size

    data_dir = os.path.join(current_dir, "..", "data/couplet", "train")
    tr_handler = preprocess(
                os.path.join(data_dir, "in.txt"), 
                os.path.join(data_dir, "out.txt"), 
                max_features, 
                maxlen
            )

    tr_handler.preprocess()
    tr_x, tr_y = tr_handler.gen_all()
    tr_y = tr_y.reshape(*tr_y.shape, 1)
    # model setting
    model = Sequential()
    # model.add(Masking(mask_value=0))
    model.add(Embedding(max_features+4, embedding_dim, input_length=maxlen+2))
    model.add(
            SimpleSeq2Seq(
                input_length=maxlen+2,
                input_dim=embedding_dim, 
                hidden_dim=256, 
                output_length=maxlen+2, 
                output_dim=embedding_dim,
                depth=2,
                dropout=0.2)
        )
    model.add(TimeDistributed(Dense(max_features+4, activation='softmax')))
    masked_categorical_crossentropy = get_loss(10003)
    model.compile(
        optimizer='rmsprop',
        loss=masked_categorical_crossentropy, 
        metrics=['accuracy'])

    plot_model(model, to_file='model.png', show_shapes=True)
    # training
    out_batch = NBatchLogger(display=100, data_handler=tr_handler)
    model.fit(tr_x, tr_y, epochs=100000, batch_size=64, callbacks=[out_batch])

    print(model.predict(tr_x[:1])[0])



    # # testing
    # model.evaluate(te_x, te_y, batch_size=32)

    # # predict
    # predicted = model.predict(te_x, batch_size=32)

    # print(te_x[0], predicted[0])



from keras.models import Model
from keras.layers import Input, LSTM, Dense, GRU, RepeatVector
def app_2():
    batch_size = 32  # Batch size for training.
    epochs = 100  # Number of epochs to train for.
    latent_dim = 50  # Latent dimensionality of the encoding space.
    num_samples = 5000  # Number of samples to train on.
    # Path to the data txt file on disk.
    data_dir = os.path.join(current_dir, "..", "data/app_1", "train")

    # Vectorize the data.
    input_texts = []
    target_texts = []
    input_characters = set()
    target_characters = set()
    with open(os.path.join(data_dir, "in.txt"), 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
    for line in lines[: min(num_samples, len(lines) - 1)]:
        input_text = line.strip()
        input_texts.append(input_text)
        for char in input_text.split(" "):
            if char not in input_characters:
                input_characters.add(char)

    with open(os.path.join(data_dir, "out.txt"), 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
    for line in lines[: min(num_samples, len(lines) - 1)]:
        target_text = line.strip()
        # We use "tab" as the "start sequence" character
        # for the targets, and "\n" as "end sequence" character.
        target_text = '\t ' + target_text + ' \n'
        target_texts.append(target_text)

        for char in target_text.split(" "):
            if char not in target_characters:
                target_characters.add(char)

    input_characters = sorted(list(input_characters))
    target_characters = sorted(list(target_characters))
    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)
    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    print('Number of samples:', len(input_texts))
    print('Number of unique input tokens:', num_encoder_tokens)
    print('Number of unique output tokens:', num_decoder_tokens)
    print('Max sequence length for inputs:', max_encoder_seq_length)
    print('Max sequence length for outputs:', max_decoder_seq_length)

    input_token_index = dict(
        [(char, i) for i, char in enumerate(input_characters)])
    target_token_index = dict(
        [(char, i) for i, char in enumerate(target_characters)])

    encoder_input_data = np.zeros(
        (len(input_texts), max_encoder_seq_length, num_encoder_tokens),
        dtype='float32')
    decoder_input_data = np.zeros(
        (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
        dtype='float32')
    decoder_target_data = np.zeros(
        (len(input_texts), max_decoder_seq_length, num_decoder_tokens),
        dtype='float32')

    for i, (input_text, target_text) in enumerate(zip(input_texts, target_texts)):
        for t, char in enumerate(input_text.split(" ")):
            encoder_input_data[i, t, input_token_index[char]] = 1.
        for t, char in enumerate(target_text.split(" ")):
            # decoder_target_data is ahead of decoder_input_data by one timestep
            decoder_input_data[i, t, target_token_index[char]] = 1.
            if t > 0:
                # decoder_target_data will be ahead by one timestep
                # and will not include the start character.
                decoder_target_data[i, t - 1, target_token_index[char]] = 1.

    print(encoder_input_data)
    # model = lstm_s2s(num_encoder_tokens, num_decoder_tokens, latent_dim)
    # Define an input sequence and process it.
    encoder_inputs = Input(shape=(None, num_encoder_tokens))
    encoder = LSTM(latent_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None, num_decoder_tokens))
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                         initial_state=encoder_states)
    decoder_dense = Dense(num_decoder_tokens, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
    model.fit([encoder_input_data, decoder_input_data], decoder_target_data,
          batch_size=batch_size,
          epochs=epochs,
          validation_split=0.2)


    # Define sampling models
    encoder_model = Model(encoder_inputs, encoder_states)

    decoder_state_input_h = Input(shape=(latent_dim,))
    decoder_state_input_c = Input(shape=(latent_dim,))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(
        decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model(
        [decoder_inputs] + decoder_states_inputs,
        [decoder_outputs] + decoder_states)

    # Reverse-lookup token index to decode sequences back to
    # something readable.
    reverse_input_char_index = dict(
        (i, char) for char, i in input_token_index.items())
    reverse_target_char_index = dict(
        (i, char) for char, i in target_token_index.items())


    def decode_sequence(input_seq):
        # Encode the input as state vectors.
        states_value = encoder_model.predict(input_seq)

        # Generate empty target sequence of length 1.
        target_seq = np.zeros((1, 1, num_decoder_tokens))
        # Populate the first character of target sequence with the start character.
        target_seq[0, 0, target_token_index['\t']] = 1.

        # Sampling loop for a batch of sequences
        # (to simplify, here we assume a batch of size 1).
        stop_condition = False
        decoded_sentence = ''
        while not stop_condition:
            output_tokens, h, c = decoder_model.predict(
                [target_seq] + states_value)

            # Sample a token
            sampled_token_index = np.argmax(output_tokens[0, -1, :])
            sampled_char = reverse_target_char_index[sampled_token_index]
            decoded_sentence += sampled_char

            # Exit condition: either hit max length
            # or find stop character.
            if (sampled_char == '\n' or
               len(decoded_sentence) > max_decoder_seq_length):
                stop_condition = True

            # Update the target sequence (of length 1).
            target_seq = np.zeros((1, 1, num_decoder_tokens))
            target_seq[0, 0, sampled_token_index] = 1.

            # Update states
            states_value = [h, c]

        return decoded_sentence


    for seq_index in range(100):
        # Take one sequence (part of the training set)
        # for trying out decoding.
        input_seq = encoder_input_data[seq_index: seq_index + 1]
        decoded_sentence = decode_sequence(input_seq)
        print('-')
        print('Input sentence:', input_texts[seq_index])
        print('Decoded sentence:', decoded_sentence)

if __name__ == '__main__':
    couplet()



