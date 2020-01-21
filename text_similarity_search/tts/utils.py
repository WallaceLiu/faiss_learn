from typing import Sequence

import numpy as np


def chunks(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def flatten_result(result: Sequence[Sequence[int]]) -> np.ndarray:
    flatten = []
    for i in range(len(result)):
        for j in range(len(result[i])):
            flatten.append((i, result[i][j]))

    flatten = np.array(flatten, dtype=np.int)

    return flatten


def save_list(lst, filepath):
    with open(filepath, 'w', encoding='utf8') as f:
        for item in lst:
            f.write(item)
            f.write('\n')


def read_list(filepath):
    with open(filepath, encoding='utf8') as f:
        rows = [l.rstrip() for l in f]
        return rows
