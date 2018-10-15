import os
import argparse
from data_process import DataHelper, Lemmatizer
from dnn import QaDNN
from flask import Flask, request, jsonify
import json
import tensorflow as tf

app = Flask(__name__)
dnn_model = QaDNN(load_from_file="QA_dnn.h5")
graph = tf.get_default_graph()

data_helper = DataHelper()
data_helper.load_vocab("QA_vocab.pkl")

lemmatizer = Lemmatizer()

@app.route("/predict",methods=['POST'])
def predict():
    global graph
    with graph.as_default():
        data = request.get_data().decode('utf-8')
        json_re = json.loads(data)
        questions = [lemmatizer.process_sentence(sentence) for sentence in json_re['questions']]
        answers = [lemmatizer.process_sentence(sentence) for sentence in json_re['answers']]
        questions = data_helper.word2index(questions)
        answers = data_helper.word2index(answers)
        predicted = [1 if prob[0] > 0.5 else 0 for prob in dnn_model.predict(questions, answers)]
        return jsonify(predicted)

@app.route("/rank",methods=['POST'])
def rank():
    global graph
    with graph.as_default():
        data = request.get_data().decode('utf-8')
        json_re = json.loads(data)
        result = []
        for question, candidates in zip(json_re['questions'], json_re['answers']):
            question = lemmatizer.process_sentence(question)
            question = data_helper.word2index(question)
            candidates = [lemmatizer.process_sentence(sentence) for sentence in candidates]
            candidates = data_helper.word2index(candidates)
            rank = [item[0] for item in dnn_model.rank(question, candidates)]
            result.append(rank)
        return jsonify(result)

if __name__ == '__main__':
    app.run(debug=False)