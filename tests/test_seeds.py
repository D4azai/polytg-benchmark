import random

import numpy as np

from polytg.training.seeds import set_global_seed


def test_seed_determinism() -> None:
    set_global_seed(91)
    first = (random.random(), np.random.random())
    set_global_seed(91)
    second = (random.random(), np.random.random())
    assert first == second

