from typing import List
import logging
import numpy as np

from source.train.NaiveBayesTrainer import TrainingResult, NaiveBayesClassifier


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
        logging.info('')
        logging.info('asd')

        corr_pred_counter = 0

        for item in data:
            input = item[0:2]
            label = int(item[2])
            result = int(float(self._predictor.predict(input)))

            make exact evaluation here!

            if result == 1:
                print(label)

            corr_pred_counter += int(result == label)

        print(corr_pred_counter)



