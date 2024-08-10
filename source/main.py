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
    dataset_evaluation = create_labeled_dataset(1500)

    visualise_data(dataset_training)

    x1_bins = [-1e-7, 0.2, 0.4, 0.6, 0.8, 1.0]
    x2_bins = [-1e-7, 0.2, 0.4, 0.6, 0.8, 1.0]

    classifier = NaiveBayesClassifier(x1_bins, x2_bins)
    classifier.learn(dataset_training)
    training_result = classifier.get_training_result()

    visualise_data(dataset_evaluation)

    evaluator = NaiveBayesEvaluator(x1_bins, x2_bins, training_result)
    evaluator.evaluate(dataset_evaluation)


if __name__ == '__main__':
    main()
