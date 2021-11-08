""" This module parses midi files for training, then trains the module for use within the predict module. """
import glob
import pickle
import numpy
from music21 import converter, instrument, note, chord
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LSTM
from keras.layers import Activation
from keras.layers import BatchNormalization as BatchNorm
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

def train_network():
    """ Trains the neural network to generate MIDI files """
    notes = get_notes()

    # get amount of pitch names in training set
    n_vocab = len(set(notes))

    #network input and output equal to prepare sequence, using notes and pitch names
    network_input, network_output = prepare_sequences(notes, n_vocab)

    #model is equal to create network function, using network input and amount of pitch names in training set
    model = create_network(network_input, n_vocab)

    #trains the algorithm using model, network input and network output
    train(model, network_input, network_output)

def get_notes():
    """ Get all the notes and chords from the midi files in the ./midi_songs directory """
    #stores notes in an array
    notes = []
    #selects all midi files from midi_songs folder
    for file in glob.glob("midi_songs/*.mid"):
        midi = converter.parse(file)
        
        #confirms how much of file is parsed so far
        print("Parsing %s" % file)

        notes_to_parse = None

        try: 
            # file has instrument parts, parses through each note in the file
            s2 = instrument.partitionByInstrument(midi)
            notes_to_parse = s2.parts[0].recurse() 
        except: 
            # file has notes in a flat structure, if instrument not found parses flat notes
            notes_to_parse = midi.flat.notes

        
        for element in notes_to_parse:
            #if single note, appends to specific pitch
            if isinstance(element, note.Note):
                notes.append(str(element.pitch))
            #if chord, combines them together to create chord instead of single note.    
            elif isinstance(element, chord.Chord):
                notes.append('.'.join(str(n) for n in element.normalOrder))
    #dumps notes into data folder            
    with open('data/notes', 'wb') as filepath:
        pickle.dump(notes, filepath)

    return notes

def prepare_sequences(notes, n_vocab):
    """ Prepare the sequences used by the Neural Network, converts notes to integer values for use in LSTM """
    sequence_length = 100

    # get all pitch names from notes
    pitchnames = sorted(set(item for item in notes))

     # create a dictionary to map pitches to integers
    note_to_int = dict((note, number) for number, note in enumerate(pitchnames))

    #creates array for network input and output
    network_input = []
    network_output = []

    # create input sequences and the corresponding outputs
    for i in range(0, len(notes) - sequence_length, 1):
        sequence_in = notes[i:i + sequence_length]
        sequence_out = notes[i + sequence_length]
        network_input.append([note_to_int[char] for char in sequence_in])
        network_output.append(note_to_int[sequence_out])

    #amount of patterns is equal to the length of network input
    n_patterns = len(network_input)

    # reshape the input into a format compatible with LSTM layers using numpy
    network_input = numpy.reshape(network_input, (n_patterns, sequence_length, 1))
    # normalize input
    network_input = network_input / float(n_vocab)
    #network output is compiled with crossentropy
    network_output = np_utils.to_categorical(network_output)

    #returns network input and output
    return (network_input, network_output)

def create_network(network_input, n_vocab):
    """ creates the structure of the LSTM Neural network  """
    model = Sequential()
    #defines 3 LSTM layers, 2 dense layers and 2 activation layers. 512 nodes used, network shape defined
    model.add(LSTM(
        512,
        input_shape=(network_input.shape[1], network_input.shape[2]),
        recurrent_dropout=0.3,
        return_sequences=True
    ))
    model.add(LSTM(512, return_sequences=True, recurrent_dropout=0.3,))
    model.add(LSTM(512))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(256))
    model.add(Activation('relu'))
    model.add(BatchNorm())
    model.add(Dropout(0.3))
    model.add(Dense(n_vocab))
    model.add(Activation('softmax'))
    #sparse categorical cross entropy used as saves time when computing, uses integers instead of vectors
    model.compile(loss='sparse_categorical_crossentropy', optimizer='rmsprop')

    #returns the model
    return model


def train(model, network_input, network_output):
    """ train the neural network, outputs a hdf5 file which can be used for running predict.py """
    #HDF5 file updated at each checkpoint when training, saves a new HDF5 everytime an improved file is created
    filepath = "weights-improvement-{epoch:02d}-{loss:.4f}-bigger.hdf5"
    checkpoint = ModelCheckpoint(
        filepath,
        monitor='loss',
        verbose=0,
        save_best_only=True,
        mode='min'
    )
    callbacks_list = [checkpoint]
    
    #200 epochs with a batch size of 128 using network input and output.
    model.fit(network_input, network_output, epochs=200, batch_size=128, callbacks=callbacks_list)

#triggers all functions to be ran
if __name__ == '__main__':
    train_network()
