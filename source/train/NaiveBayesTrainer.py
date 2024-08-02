import numpy as np
from typing import List, Dict, Optional
import pandas as pd
from dataclasses import dataclass
import logging
import operator

from source.data.DataGenerator import DataGenerator


@dataclass
class TrainingResult:
    class_probs_dict: Dict[str, float]
    cond_probs: np.ndarray


class NaiveBayesClassifier:

    def __init__(self, x1_bins: List[float], x2_bins: List[float]):
        self._x1_bins = x1_bins
        self._x2_bins = x2_bins

        self._trainer = NaiveBayesTrainer
        self._training_result = None

    def learn(self, data_arr_labeled: np.ndarray):
        self._training_result = self._trainer.train(data_arr_labeled, self._x1_bins, self._x2_bins)

    def predict(self, x: np.ndarray) -> Optional[str]:
        key_max_log_prob = None
        if self._training_result:
            logging.info('make prediction')
            cell_index = DataGenerator.get_cell_index(x, self._x1_bins, self._x2_bins)
            x_encoded = DataGenerator.encode_data_sample(cell_index[0], cell_index[1],
                                                         len(self._x1_bins) - 1, len(self._x2_bins) - 1)
            log_probs_dict = {}
            for i, item in enumerate(self._training_result.class_probs_dict.keys()):
                v1 = np.multiply(x_encoded, np.log(self._training_result.cond_probs[:, :, i].ravel())).reshape(-1, 1)
                v2 = np.multiply(1 - x_encoded,
                                 np.log(1 - self._training_result.cond_probs[:, :, i].ravel())).reshape(-1, 1)

                log_probs_dict[item] = sum(v1 + v2) + np.log(self._training_result.class_probs_dict[item])
            key_max_log_prob = max(log_probs_dict.items(), key=operator.itemgetter(1))[0]
        else:
            logging.info('can`t make prediction - there is no training result available')
        return key_max_log_prob

    def get_training_result(self) -> TrainingResult:
        return self._training_result

    def set_training_result(self, training_result: TrainingResult):
        self._training_result = TrainingResult(training_result.class_probs_dict.copy(),
                                               training_result.cond_probs.copy())

class NaiveBayesTrainer:
    @staticmethod
    def additive_smoothing(absolute_frequency: int, tot_num_events, num_features: int, pseudocount: float = 1) -> float:
        logging.info('apply additive smoothing')
        return (absolute_frequency + pseudocount) / (tot_num_events + pseudocount * num_features)

    @staticmethod
    def compute_class_probabilities(data_arr_labeled: np.ndarray, class_labels: List[int]) -> Dict[str, float]:
        logging.info('compute class probabilities')
        num_samples = len(data_arr_labeled)
        class_probs_dict = {}
        for c in class_labels:
            class_probs_dict[str(c)] = len([item for item in data_arr_labeled if item[2] == c]) / num_samples
        return class_probs_dict

    @staticmethod
    def compute_conditional_probabilities(data: pd.DataFrame, num_labels: int, num_cells_x1: int, num_cells_x2: int,
                                          apply_additive_smoothing: bool = True) -> np.ndarray:
        logging.info('compute conditional probabilities')
        cond_probs = np.zeros((num_cells_x1, num_cells_x2, num_labels))
        for k in range(0, num_labels):
            num_samples_class = np.array(data.iloc[:, 2] == k, dtype=np.uint8).sum()
            for i1 in range(0, num_cells_x1):
                for i2 in range(0, num_cells_x2):
                    df_filtered = data.query('i1 == @i1 & i2 == @i2 & label == @k')
                    absolute_frequency = len(df_filtered)
                    if apply_additive_smoothing:
                        cond_probs[i1, i2, k] = NaiveBayesTrainer.additive_smoothing(absolute_frequency,
                                                                                     num_samples_class,
                                                                                     num_cells_x1 * num_cells_x2)
                    else:
                        cond_probs[i1, i2, k] = absolute_frequency / num_samples_class
        return cond_probs

    @staticmethod
    def train(data_arr_labeled: np.ndarray, x1_bins, x2_bins) -> TrainingResult:
        logging.info('train naive bayes classifier')

        # compute class probabilities/relative frequencies
        class_labels = list(set(data_arr_labeled[:, 2]))
        class_probs_dict = NaiveBayesTrainer.compute_class_probabilities(data_arr_labeled, class_labels)

        # categorise data
        data_categorised = DataGenerator.categorise_data(data_arr_labeled, x1_bins, x2_bins)

        #   > a one hot encoding vector is generated and used as representation for the discretised
        #       data sample; it is a vector of n1 * n2 elements containing zeros, except a 1 at index i1 * n2 + i2

        # compute conditional probabilities using discretised data
        cond_probs = NaiveBayesTrainer.compute_conditional_probabilities(data_categorised, len(class_labels),
                                                                         len(x1_bins) - 1, len(x2_bins) - 1)
        return TrainingResult(class_probs_dict, cond_probs)
