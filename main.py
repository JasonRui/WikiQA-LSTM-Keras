import os
import argparse
from data_process import DataHelper
from dnn import QaDNN
import pickle

'''
parse argument
'''
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--embedding_file', action='store', help='embedding file path')
    parser.add_argument('--data', action='store', help='directory to train/dev/test data')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    # args.data = "data"
    # args.embedding_file = "glove.6B.50d.txt"
    data_helper = DataHelper()
    data_helper.build(args.embedding_file, args.data + "/WikiQA-train.csv", args.data + "/WikiQA-dev.csv", args.data + "/WikiQA-test.csv")

    questions = [item[0] for item in data_helper.train_data]
    answers = [item[1] for item in data_helper.train_data]
    labels = [item[2] for item in data_helper.train_data]

    val_questions = [item[0] for item in data_helper.dev_data]
    val_answers = [item[1] for item in data_helper.dev_data]
    val_labels = [item[2] for item in data_helper.dev_data]
    dnn_model = QaDNN(
        questions=questions,
        answers=answers,
        labels=labels,
        val_questions=val_questions,
        val_answers=val_answers,
        val_labels=val_labels,
        word_embeddings=data_helper.word_embeddings,
        word_embeddings_trainable=True
    )
    dnn_model.train()
    dnn_model.save_model("QA_dnn.h5")
    data_helper.save_vocab("QA_vocab.pkl")
    mrr = dnn_model.evaluate(data_helper.test_data)
    print("MRR: ", mrr)
