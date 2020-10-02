import numpy as np

def entropy(ps):
    '''Compute entropy.'''
    ps.flatten()
    ps = ps[np.nonzero(ps)]            # toss out zeros
    H = -np.sum(ps * np.log2(ps))   # compute entropy
    
    return H

def density(vec_list):
    assert type(vec_list) is list or len(vec_list.shape) > 1 
    ids, counts = np.unique(np.concatenate([vec.reshape(-1,1) for vec in vec_list], axis=1), axis=0, return_counts = True)
    return counts / np.sum(counts)


def mi(x, y):
    '''Compute mutual information'''
    p_xy = density([x, y])
    p_x = density([x])
    p_y = density([y])
    
    H_xy = entropy(p_xy)
    H_x  = entropy(p_x)
    H_y  = entropy(p_y)
    
    return H_x + H_y - H_xy

def con_mi(x, y, z):
    p_xyz = density([x, y, z])
    p_xz = density([x, z])
    p_yz = density([y, z])
    p_z = density([z])

    return entropy(p_xz) + entropy(p_yz) - entropy(p_xyz) - entropy(p_z)
    
def complexity(A, B): #A,B are matrices
    return np.array([mi(A[i], B[i]) for i in range(len(A))])

def complexity_average(A, B):
    return mi(A, B)

def complexity_cmi(A, B, C):
    return np.array([con_mi(A[i], B[i], C[i]) for i in range(len(A))])

def complexity_average_cmi(A, B, C):
    return con_mi(A, B, C)
