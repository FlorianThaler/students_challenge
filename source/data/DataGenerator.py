import numpy as np
import pandas as pd
from typing import Callable, List
import logging


class DataGenerator:

    @staticmethod
    def draw_sample_from_distribution(f1: Callable, f2: Callable, sample_size: int) -> np.ndarray:
        # NOTE
        #   > we consider only distributions on [0, 1] x [0, 1] induced by
        #     a random variable (f1(X1), f2(X2)), where
        #       * X1, X2 ~ Uni(0, 1)
        #       * f1, f2:[0, 1] -> [0, 1] measurable
        x1 = np.random.uniform(0, 1, sample_size)
        x2 = np.random.uniform(0, 1, sample_size)

        return np.vstack((f1(x1), f2(x2))).T

    @staticmethod
    def label_data_by_subset(data: np.ndarray, indicator_func: Callable) -> np.ndarray:
        n = len(data)
        labels = np.zeros(n, dtype=np.int8)
        for i in range(0, n):
            labels[i] = int(indicator_func(data[i, :]))
        return labels

    @staticmethod
    def create_data(sample_size: int) -> np.ndarray:
        f1 = lambda x: 0.5 * (1 + np.cos(np.pi * x))
        f2 = lambda x: np.sin(np.pi * x)
        data_arr = DataGenerator.draw_sample_from_distribution(f1, f2, sample_size)

        # TODO
        #   > make subset configurable !!!
        indicator_func = lambda x: np.linalg.norm(x - np.array([0.5, 0.5]), 1) <= 0.25

        labels = DataGenerator.label_data_by_subset(data_arr, indicator_func)
        data_arr_labeled = np.append(data_arr, labels.reshape(len(data_arr), 1), axis=1)

        return data_arr_labeled

    @staticmethod
    def get_cell_index(data_sample: np.ndarray, x1_bins: List[float], x2_bins: List[float]) -> np.ndarray:
        i1 = np.digitize(data_sample[1], x2_bins) - 1
        i2 = np.digitize(data_sample[0], x1_bins) - 1
        return np.array([i1, i2], dtype=np.uint16)

    @staticmethod
    def categorise_data(data_arr_labeled: np.ndarray, x1_bins: List[float], x2_bins: List[float]) -> pd.DataFrame:
        # NOTE
        #   > data is sampled w.r.t. the coordinate system
        #           ^ x2
        #           |
        #           |
        #           |
        #           |
        #           -------------> x1
        #   > a data sample (x1, x2) is categorised as follows: let x1_bins, x2_bins
        #       denote lists of bins along the x1-axis and the x2-axis respectively.
        #       let i1, i2 such that
        #           x1_bins[i1] < x2 <= x1_bins[i1 + 1] and x2_bins[i2] < x1 <= x2_bins[i2 + 1]
        logging.info('categorise continuous data')
        cell_index_list = []
        for item in data_arr_labeled:
            cell_index_list.append(DataGenerator.get_cell_index(item[0 : 2], x1_bins, x2_bins))
        cell_index_array = np.array(cell_index_list)

        df = pd.DataFrame({'x1': data_arr_labeled[:, 0],
                           'x2': data_arr_labeled[:, 1],
                           'label': data_arr_labeled[:, 2],
                           'i1': cell_index_array[:, 0],
                           'i2': cell_index_array[:, 1]})

            # ### debugging ...
            # cell_counts = DataGenerator.cell_counts_from_categorised_data(x1_bins, x2_bins, df)

        return df

    @staticmethod
    def cell_counts_from_categorised_data(x1_bins: List[float], x2_bins: List[float],
                                          df: pd.DataFrame) -> List[np.ndarray]:
        labels = list(set(df['label'].tolist()))
        cell_array_list = []
        for item in labels:
            cell_array = np.zeros((len(x2_bins) - 1, len(x1_bins) - 1), dtype=np.uint16)
            for index, row in df.iterrows():
                if int(row['label']) == int(item):
                    cell_array[int(row['i2']), int(row['i1'])] += 1

            cell_array_list.append(cell_array)
        return cell_array_list

    @staticmethod
    def encode_data_sample(i1: int, i2: int, n1: int, n2: int) -> np.ndarray:
        ret_val = np.zeros(n1 * n2)
        ret_val[i1 * n2 + i2] = 1
        return ret_val
