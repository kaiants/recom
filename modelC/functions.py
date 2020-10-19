import numpy as np

def rand_argmin(vec: list) -> int:
    return np.random.choice(np.flatnonzero(vec == vec.min()))


def rand_argmax(vec: list) -> int:
    return np.random.choice(np.flatnonzero(vec == vec.max()))

    
def lowrank_approx(a,rank):
    u, s, vh = np.linalg.svd(a)
    u = u[:,:rank]
    vh = vh[:rank,:]
    s = s[:rank]
    s = np.diag(s)
    low_rank = np.dot(np.dot(u,s),vh)
    return low_rank

def compute_second_eigenvector(a):
    u, s, vh = np.linalg.svd(a,  hermitian=True)
    return u[:,1]
    
    