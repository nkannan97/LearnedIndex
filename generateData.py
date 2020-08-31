import numpy as np
import os
import sys
from enum import Enum
import random
import csv


class Distribution(Enum):
    NORMAL = 1
    LOGNORMAL = 2
    UNIFORM = 3
    RANDOM = 4

    @classmethod
    def to_string(cls, val):
        for k, v in vars(cls).items():
            if v == val:
                return k.lower()


def get_data(distribution, size):
    data = []
    if distribution == Distribution.NORMAL:
        data = np.random.normal(1000, 100, size)
    elif distribution == Distribution.LOGNORMAL:
        data = np.random.lognormal(0, 2, size)
    elif distribution == Distribution.RANDOM:
        data = random.sample(range(size*2), size)
    else:
        data = np.random.uniform(0, 100, size)

    return data


def get_sorted_data(distribution, size):
    data = get_data(distribution, size)
    data.sort()
    return data


def generate_data(distribution, size):
    generateData = get_sorted_data(distribution, size)
    data_path = os.path.join(
        os.getcwd(), Distribution.to_string(distribution) + ".csv")
    with open(data_path, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        for index, val in enumerate(generateData):
            csv_writer.writerow([val])


if __name__ == "__main__":
    data_size = int(sys.argv[1]) if len(sys.argv) == 2 else 1000
    generate_data(Distribution.NORMAL, data_size)
    generate_data(Distribution.LOGNORMAL, data_size)
    generate_data(Distribution.UNIFORM, data_size)
    generate_data(Distribution.RANDOM, data_size)
