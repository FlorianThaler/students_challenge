import numpy as np
import logging

from data.DataGenerator import DataGenerator
from data.utils import visualise_data
from train.NaiveBayesTrainer import NaiveBayesClassifier
from evaluate.NaiveBayesEvaluator import NaiveBayesEvaluator


def seed_packages(seed_val=666):
    logging.info('seed packages')
    np.random.seed(seed_val)


def create_labeled_dataset(sample_size=1000) -> np.ndarray:
    logging.info('create labeled dataset')
    data_arr_labeled = DataGenerator.create_data(sample_size)
    return data_arr_labeled


def setup_logger():
    logging.basicConfig(format='%(asctime)s - %(message)s', datefmt='%d-%b-%yT%H:%M:%S', level=logging.INFO)


def main():
    setup_logger()
    seed_packages()
    dataset_training = create_labeled_dataset()
    dataset_evaluation = create_labeled_dataset(200)

    visualise_data(dataset_training)

    x1_bins = [-1e-7, 0.2, 0.4, 0.6, 0.8, 1.0]
    x2_bins = [-1e-7, 0.2, 0.4, 0.6, 0.8, 1.0]

    classifier = NaiveBayesClassifier(x1_bins, x2_bins)
    classifier.learn(dataset_training)
    training_result = classifier.get_training_result()

    evaluator = NaiveBayesEvaluator(x1_bins, x2_bins, training_result)
    evaluator.evaluate(dataset_evaluation)

    #
    #
    # x = np.array([0.21, 0.87])
    # # x = np.array([0.5, 0.5])
    #
    # result = classifier.predict(x)
    # logging.info('# ### ###############################################################')
    # logging.info('classified ({:f}, {:f}) as: {:s}'.format(x[0], x[1], result))


if __name__ == '__main__':
    main()
