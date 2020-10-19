import numpy as np

def rand_argmin(vec: list) -> int:
    return np.random.choice(np.flatnonzero(vec == vec.min()))


def rand_argmax(vec: list) -> int:
    return np.random.choice(np.flatnonzero(vec == vec.max()))

def kl_inv_approx(b: float, val: float) -> float:
    # return a, with kl(a,b) aproximately equal to val and a <= b
    # use　formula kl(a+eps,a) ~ eps^2 /(2 a (1-a)) 
    eps = np.sqrt(val * b * (1 - b) * 2 )
    return b - eps

    
def kl(a, b):
    return a * np.log(a/b) + (1-a)*np.log((1-a)/(1-b))
    
    