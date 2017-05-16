import libitg as libitg
from libitg import CFG
import numpy as np
import pickle

def write_derrivation(d):
    derivation_as_fsa = libitg.forest_to_fsa(CFG(d), d[0].lhs)
    candidates = libitg.enumerate_paths_in_fsa(derivation_as_fsa)
    return candidates

def joint_prob(derrivation, estimated_weights, inside, root, log=False):
    """
    Computes the joint probability of a a sentence and its derrivation.
    """
    numerator = np.exp(sum([np.log(estimated_weights[edge]) for edge in derrivation]))
    Z = inside[root]
    if log: print(numerator, Z)
    return numerator / Z

def save_weights(weights, savepath):
    f = open(savepath + 'weights.pkl', 'wb')
    pickle.dump(weights, f, protocol=4)
    f.close()

def load_weights(savepath):
    print('weights loaded')
    f = open(savepath + 'weights.pkl', 'rb')
    weights = pickle.load(f)
    f.close()
    return weights