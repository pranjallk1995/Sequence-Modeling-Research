"""
module to separete features and labels from the dataset
"""

import config as cfg
import tensorflow as tf

class FeatureExtractor():
    """ class to extract features from datasets """

    def __init__(self, dataset_tensors: tf.Tensor) -> None:
        self.tensors = dataset_tensors

    def split_tensors(self) -> list[tf.Tensor, tf.Tensor]:
        """ function to extract features from given dataset """
        return [
            self.tensors[:, :cfg.MAX_FEATURE_LENGTH],
            self.tensors[:, cfg.MAX_FEATURE_LENGTH:]
        ]

    def run(self) -> list[tf.Tensor, tf.Tensor]:
        """ module entry point """
        return self.split_tensors()
