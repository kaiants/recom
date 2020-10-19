import numpy as np

def rand_argmin(vec: list) -> int:
    return np.random.choice(np.flatnonzero(vec == vec.min()))


def rand_argmax(vec: list) -> int:
    return np.random.choice(np.flatnonzero(vec == vec.max()))

    
    
    
    