import numpy as np
import datetime
import time
import pickle
import h5py
from sklearn.metrics import accuracy_score
from keras.datasets import reuters
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential, load_model, Model
from keras.layers import Input, Dense, Embedding, LSTM, Dropout, Activation, concatenate, BatchNormalization
from keras import optimizers
from keras.callbacks import Callback, ModelCheckpoint
from keras.wrappers.scikit_learn import KerasClassifier


class QaDNN(object):
    def __init__(self, questions=None, answers=None, labels=None, word_embeddings=None, word_embeddings_trainable=True, val_questions=None, val_answers=None, val_labels=None, maxlen=40, epochs=5, batch_size=30, load_from_file=None):

        self.questions = np.array(questions)
        self.answers = np.array(answers)
        self.labels = np.array(labels)

        self.val_questions = np.array(val_questions)
        self.val_answers = np.array(val_answers)
        self.val_labels = np.array(val_labels)

        self.maxlen = maxlen
        self.epochs = epochs
        self.batch_size = batch_size
        if load_from_file is not None:
            self.load_model(load_from_file)
            return
        vocab_size, embedding_size = word_embeddings.shape
        print('Shape of questions:', self.questions.shape)
        print('Shape of answers:', self.answers.shape)
        print('Shape of labels:', self.labels.shape)

        # model input
        q_input = Input(shape=(maxlen,))
        a_input = Input(shape=(maxlen,))

        # embedding layer
        embedding_layer = Embedding(vocab_size, embedding_size, weights=[word_embeddings], input_length=maxlen,
                  trainable=word_embeddings_trainable)

        # convert question and answer input to word embedding matrix separately
        q = embedding_layer(q_input)
        a = embedding_layer(a_input)

        # add LSTM layer, get their new representation
        q = LSTM(50, return_sequences=True)(q)
        a = LSTM(50, return_sequences=True)(a)

        q = LSTM(50, return_sequences=False)(q)
        a = LSTM(50, return_sequences=False)(a)

        # concatenate those two vectors for classification
        merged = concatenate([q, a])

        # add dense layer, and dropout, batch normalization to prevent over fitting
        merged = Dense(200, activation='relu')(merged)
        merged = Dropout(0.2)(merged)
        merged = BatchNormalization()(merged)

        # another same layer
        merged = Dense(200, activation='relu')(merged)
        merged = Dropout(0.2)(merged)
        merged = BatchNormalization()(merged)

        # output the probability
        output = Dense(1, activation='sigmoid')(merged)

        self.model = Model(inputs=[q_input, a_input], output=output)

        # use cross entropy as loss function and Adam to optimize
        adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=False)
        self.model.compile(loss='binary_crossentropy', optimizer=adam)

    def load_model(self, path):
        self.model = load_model(path)

    def save_model(self, path):
        self.model.save(path)


    def train(self):
        # padding input first
        questions_padded = pad_sequences(self.questions, maxlen=self.maxlen, padding='post')
        answers_padded = pad_sequences(self.answers, maxlen=self.maxlen, padding='post')

        # get validation data
        validation_data = None
        if not self.val_questions is None:
            val_questions_padded = pad_sequences(self.val_questions, maxlen=self.maxlen, padding='post')
            val_answers_padded = pad_sequences(self.val_answers, maxlen=self.maxlen, padding='post')
            validation_data = ([val_questions_padded, val_answers_padded], [self.val_labels])

        print("Starting training at", datetime.datetime.now)
        t0 = time.time()
        callbacks = [ModelCheckpoint("model/question_answer_weights.h5", monitor='val_loss', save_best_only=True)]
        # as our data for binary classification is skewed, give more weights to class with less samples
        class_weight = {0: 1.,
                        1: 6.}
        # start training
        history = self.model.fit([questions_padded, answers_padded], [self.labels], epochs=self.epochs, batch_size=self.batch_size, callbacks=callbacks,
                                 validation_data=validation_data, class_weight=class_weight)
        t1 = time.time()
        print("Training ended at", datetime.datetime.now())
        print("Minutes elapsed: %f" % ((t1 - t0) / 60.))

        min_val_loss, idx = min((val, idx) for (idx, val) in enumerate(history.history['val_loss']))
        print("Minimum validation loss = {0:.4f} (epoch {1:d})".format(min_val_loss, idx + 1))

    '''
    predict whether a question and answer match
    '''
    def predict(self, questions, answers):
        questions_padded = pad_sequences(questions, maxlen=self.maxlen, padding='post')
        answers_padded = pad_sequences(answers, maxlen=self.maxlen, padding='post')
        return self.model.predict([questions_padded, answers_padded])

    '''
    give a question with multi candidate answers, rank those answers
    '''
    def rank(self, question, answers):
        questions = np.tile(question, (len(answers), 1))
        scores = [item[0] for item in self.predict(questions, answers)]
        sorted_answers = sorted(enumerate(scores), key=lambda k:k[1], reverse=True)
        return sorted_answers

    '''
    calculate Mean Reciprocal Rank (MRR) score
    '''
    def evaluate(self, test_data):
        MRR = 0
        count = 0
        for question, candidates, labels in test_data:
            count += 1
            true_answer = labels.index(1)
            rank = self.rank(question, candidates)
            ranking = [item[0] for item in rank].index(true_answer)
            MRR += 1 / (ranking + 1)
        return MRR / count


