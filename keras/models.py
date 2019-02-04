'''
Based on farizrahman4u/seq2seq
- pip3 install git+https://github.com/datalogai/recurrentshop.git 
- pip3 install git+https://github.com/farizrahman4u/seq2seq.git
'''
import numpy as np

import seq2seq
from seq2seq.models import SimpleSeq2Seq

def simple_s2s(input_length, input_dim, output_length, output_dim, hidden_dim, depth):
    model = SimpleSeq2Seq(
                input_shape=(input_length, input_dim), 
                hidden_dim=hidden_dim, 
                output_length=output_length, 
                output_dim=output_dim, 
                depth=depth
            )
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model



from keras.models import Model
from keras.layers import Input, LSTM, Dense

def lstm_s2s(input_length, output_length, hidden_dim):
    encoder_inputs = Input(shape=(None, input_length))
    encoder = LSTM(hidden_dim, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    # We discard `encoder_outputs` and only keep the states.
    encoder_states = [state_h, state_c]

    # Set up the decoder, using `encoder_states` as initial state.
    decoder_inputs = Input(shape=(None, output_length))
    # We set up our decoder to return full output sequences,
    # and to return internal states as well. We don't use the
    # return states in the training model, but we will use them in inference.
    decoder_lstm = LSTM(hidden_dim, return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs,
                                         initial_state=encoder_states)
    decoder_dense = Dense(output_length, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)

    # Define the model that will turn
    # `encoder_input_data` & `decoder_input_data` into `decoder_target_data`
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop')

    return model
