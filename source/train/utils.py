import numpy as np
from typing import List



def train_naive_bayes_classifier(data_arr_labeled: np.ndarray):

    labels = list(set([item[2] for item in data_arr_labeled]))
    labels.sort()

    class_probs = compute_class_probabilities(data_arr_labeled, labels)
    cond_probs = compute_conditional_probabilities(data_arr_labeled)

    return class_probs, cond_probs