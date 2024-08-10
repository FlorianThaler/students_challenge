from typing import List
import logging
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from dataclasses import dataclass

from source.train.NaiveBayesTrainer import TrainingResult, NaiveBayesClassifier


@dataclass
class EvaluationMetricItem:
    name: str
    value: float


class NaiveBayesEvaluator:

    def __init__(self, x1_bins: List[float], x2_bins: List[float], training_result: TrainingResult):

        self._predictor = NaiveBayesClassifier(x1_bins, x2_bins)
        self._predictor.set_training_result(training_result)

        self._make_dummy_prediction()

    def _make_dummy_prediction(self):
        logging.info('make dummy prediction')
        try:
            x = np.random.random(2)
            self._predictor.predict(x)
        except:
            raise Exception('could not make dummy prediction - check training result of classifier!')

    def evaluate(self, data: np.ndarray):
        dct_list = []
        for item in data:
            tmp_dct = {'x': item[0], 'y': item[1],
                       'groundtruth': int(item[2]),
                       'prediction': int(float(self._predictor.predict(item[0:2])))}
            dct_list.append(tmp_dct)
        df = pd.DataFrame(dct_list)

        # NOTE
        #   > there is the following mapping of labels
        #       * 0 = green
        #       * 1 = red
        #   > according to the task description the identification of data points labeled as
        #       1 is of major importante. thus we arrange the labels in such a way that
        #       a true positive classification corresponds to the situation
        #           * groundtruth = 1
        #           * prediction = 1
        #       consequently the true positive rate has to be interpreted accordingely.
        conf_mtrx = confusion_matrix(df['groundtruth'].values.tolist(),
                                     df['prediction'].values.tolist(), labels=[0, 1])

        metrics = self._compute_metrics(conf_mtrx)
        self._print_metrics(metrics)

    def _compute_metrics(self, conf_mtrx: np.ndarray) -> List[EvaluationMetricItem]:
        metrics_list = []
        metrics_list.append(EvaluationMetricItem('accuracy', self._compute_accuracy(conf_mtrx)))
        metrics_list.append(EvaluationMetricItem('true_positive_rate', self._compute_true_positive_rate(conf_mtrx)))
        metrics_list.append(EvaluationMetricItem('false_negative_rate', self._compute_false_negative_rate(conf_mtrx)))
        return metrics_list

    @staticmethod
    def _print_metrics(metrics: List[EvaluationMetricItem]):
        for item in metrics:
            logging.info('{:s}: {:.4f}'.format(item.name, item.value))

    @staticmethod
    def _compute_accuracy(conf_mtrx: np.ndarray) -> float:
        acc = -1.0
        num_samples = conf_mtrx.sum()
        if num_samples > 0:
            acc = conf_mtrx.trace() / num_samples
        return acc

    @staticmethod
    def _compute_true_positive_rate(conf_mtrx: np.ndarray) -> float:
        # NOTE
        #   > true_positive_rate = recall
        #   > confusion matrix structure in sklearn is as follows
        #                          pred
        #               | c_1 | c_2 | c_3 | ... | cn
        #       t   c_1 |
        #       r   c_2 |
        #       u   c_3 |
        #       e   ...
        #           c_n |

        tpr = -1.0
        if conf_mtrx[1, 1] > 0:
            tpr = conf_mtrx[1, 1] / conf_mtrx.sum(axis=1)[1]
        return tpr

    @staticmethod
    def _compute_false_negative_rate(conf_mtrx: np.ndarray) -> float:
        fnr = -1.0
        if conf_mtrx[1, 1] > 0:
            fnr = conf_mtrx[1, 0] / conf_mtrx.sum(axis=1)[1]
        return fnr
