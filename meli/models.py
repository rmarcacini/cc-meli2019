import keras
from keras.models import Sequential
from keras.layers import Embedding
from keras.layers import Dropout
from keras.layers import Bidirectional
from keras.layers import GRU
from keras.layers import LSTM
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.layers import Input
from keras.layers import Conv1D
from keras.layers import SpatialDropout1D
from keras.layers import MaxPool1D
from keras.layers.merge import concatenate
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPool1D
from keras.models import Model
import numpy as np
import random


def TextGRU(num_words, input_length, embedding_dim, embedding_matrix, number_of_classes):
    
    
    model = Sequential()
    model.add(Embedding(num_words, embedding_dim, input_length=input_length,trainable=random.choice([True,False]),weights=[embedding_matrix]))
    model.add(Bidirectional(GRU(embedding_dim, return_sequences=False)))
    model.add(Dense(max(3*number_of_classes,3*embedding_dim), activation='relu'))
    model.add(Dropout(random.choice([0.1,0.2,0.3])))
    model.add(Dense(max(2*number_of_classes,2*embedding_dim), activation='relu'))
    model.add(Dropout(random.choice([0.1,0.2,0.3])))
    model.add(Dense(number_of_classes, activation='softmax'))

    L=[]
    L.append(keras.optimizers.Adam(lr=random.choice([0.001,0.001,0.002,0.003]), beta_1=0.9, beta_2=0.999, amsgrad=False))
    L.append(keras.optimizers.Adadelta(lr=1.0, rho=0.95))
    opt = random.choice(L)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    model.summary()
    
    return model


def TextLSTM(num_words, input_length, embedding_dim, embedding_matrix, number_of_classes):
    
    
    model = Sequential()
    model.add(Embedding(num_words, embedding_dim, input_length=input_length,trainable=random.choice([True,False]),weights=[embedding_matrix]))
    model.add(Bidirectional(LSTM(embedding_dim, return_sequences=False)))
    model.add(Dense(max(3*number_of_classes,3*embedding_dim), activation='relu'))
    model.add(Dropout(random.choice([0.1,0.2,0.3])))
    model.add(Dense(max(2*number_of_classes,2*embedding_dim), activation='relu'))
    model.add(Dropout(random.choice([0.1,0.2,0.3])))
    model.add(Dense(number_of_classes, activation='softmax'))

    L=[]
    L.append(keras.optimizers.Adam(lr=random.choice([0.001,0.001,0.002,0.003]), beta_1=0.9, beta_2=0.999, amsgrad=False))
    L.append(keras.optimizers.Adadelta(lr=1.0, rho=0.95))
    opt = random.choice(L)
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    model.summary()
    
    return model


def TextCNN(num_words, input_length, embedding_dim, embedding_matrix, number_of_classes):
    
    sequence_input = Input(shape=(input_length,), dtype='int32')
    embedding_layer = Embedding(num_words, embedding_dim, embeddings_initializer=keras.initializers.random_uniform(minval=-0.25, maxval=0.25),input_length=input_length,trainable=random.choice([True,False]),weights=[embedding_matrix])
    embedded_sequences = embedding_layer(sequence_input)

    # create a convolution + maxpool layer for each filter size
    NUM_FILTERS = random.choice([64,128,256])
    FILTER_SIZES = [2, 3, 4]
    pooled_outputs = []
    for filter_size in FILTER_SIZES:
        x = Conv1D(NUM_FILTERS, filter_size, activation='relu')(embedded_sequences)
        x = MaxPool1D(int(x.shape[1]))(x)
        pooled_outputs.append(x)
    merged = concatenate(pooled_outputs)
    #x = Flatten()(merged)
    x = Bidirectional(GRU(embedding_dim, return_sequences=False))(x)
    x = Dense(max(3*number_of_classes,3*embedding_dim), activation='relu')(x)
    x = Dropout(random.choice([0.1,0.2,0.3]))(x)
    x = Dense(max(2*number_of_classes,2*embedding_dim), activation='relu')(x)
    x = Dropout(random.choice([0.1,0.2,0.3]))(x)
    outputs = Dense(number_of_classes, activation='softmax')(x)
    model = Model(sequence_input, outputs)

    L = []
    L.append(keras.optimizers.Adam(lr=random.choice([0.001,0.001,0.002,0.003]), beta_1=0.9, beta_2=0.999, amsgrad=False))
    L.append(keras.optimizers.Adadelta(lr=1.0, rho=0.95))
    opt = L[random.randrange(len(L))]

    model.compile(loss='categorical_crossentropy',optimizer=opt,metrics=['accuracy'])

    model.summary()
    
    return model


def data_input(df,embeddings_index):

    # The maximum number of words to be used. (most frequent)
    MAX_NB_WORDS = random.randrange(100000, 250000, 10000)
    MAX_SEQUENCE_LENGTH = int(random.randrange(10, 15))
    EMBEDDING_DIM = 300
    tokenizer = keras.preprocessing.text.Tokenizer(num_words=MAX_NB_WORDS, lower=True)
    tokenizer.fit_on_texts(df['title_clean'].apply(str))
    word_index = tokenizer.word_index
    
    words_not_found = []
    nb_words = min(MAX_NB_WORDS, len(word_index))
    embedding_matrix = np.zeros((nb_words, EMBEDDING_DIM))
    for word, i in word_index.items():
        if i >= nb_words:
            continue
        embedding_vector = embeddings_index.get(word)
        if (embedding_vector is not None) and len(embedding_vector) > 0:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
        else:
            words_not_found.append(word)
    
    return tokenizer, embedding_matrix, MAX_NB_WORDS, MAX_SEQUENCE_LENGTH, nb_words