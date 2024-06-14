""" module to preprecess data before passing to the model """

import os
import json as js
import pandas as pd
import config as cfg
import logging as lg
import tensorflow as tf

from typing import Set

class PreProcess():
    """ class to perform preprocessing on the dataset """

    def __init__(self) -> None:
        self.dataset = None
        self.dataset_id = None
        self.vocab = {}
        self.word_count = 1

    def drop_punctuations(self, sentence: str) -> str:
        """ function to remove all punctuations """

        for punctuation in cfg.PUNCTUATIONS:
            sentence = sentence.replace(punctuation, "")
        return sentence

    def build_vocab_and_vectorize(self, dataset: pd.DataFrame) -> None:
        """ function to build the vocabulary from the dataset """

        sentences = [
            self.drop_punctuations(sentence.lower()).split(" ") for sentence in dataset["Sentence"]
        ]
        for sentence_index, sentence in enumerate(sentences):
            for word_index, word in enumerate(sentence):
                if self.vocab.get(word, None) is None:
                    self.vocab[word] = self.word_count
                    self.word_count += 1
                sentences[sentence_index][word_index] = self.vocab.get(word)
            if len(sentences[sentence_index]) < cfg.MAX_INPUT_VECTOR_LENGTH:
                while len(sentences[sentence_index]) < cfg.MAX_INPUT_VECTOR_LENGTH:
                    sentences[sentence_index].append(0)
        return tf.convert_to_tensor(sentences)

    def run(
            self, dataset_id: int, dataset: pd.DataFrame, dataset_type: cfg.DatasetType
        ) -> tf.Tensor:
        """ module entry point """

        lg.info("Performing vectorization of the dataset chunk %d", dataset_id)
        dataset_tensor = self.build_vocab_and_vectorize(dataset)
        with open(
            os.path.join(cfg.TRAINING_DATA_PATH, dataset_type.value["json"]), "w", encoding="utf-8"
        ) as vocab_json:
            js.dump(self.vocab, vocab_json, indent=4)
        lg.info("dataset vocabulory stored in %s", dataset_type.value["json"])
        lg.debug("vocab size: %d", len(self.vocab))
        return dataset_tensor
