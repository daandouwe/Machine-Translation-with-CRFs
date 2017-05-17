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

def joint_prob_log(derrivation, estimated_weights, inside, root):
    """
    Computes the joint probability of a a sentence and its derrivation.

    Note: estimated_weights are logged!
    """
    numerator = sum([estimated_weights[edge] for edge in derrivation])
    Z = inside[root]
    return np.exp(numerator - Z)

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

def save_weights_log(weights, savepath):
    f = open(savepath + 'weights-log.pkl', 'wb')
    pickle.dump(weights, f, protocol=4)
    f.close()

def load_weights_log(savepath):
    print('weights loaded')
    f = open(savepath + 'weights-log.pkl', 'rb')
    weights = pickle.load(f)
    f.close()
    return weights


def check_nan_inf(I_tgt, I_ref):
    """
    Checking for nan and inf
    """
    check_nan = [np.isnan(v) for v in I_tgt.values()]
    check_nan_log = [np.isnan(np.log(v)) for v in I_tgt.values()]
#             print(check_nan_log)
#             print(sum(check_nan_log))
    if np.sum(check_nan):
        # early stopping if we get nan
        print('Early stop due to nan in I_tgt tgt_edge_weights')
        print(I_tgt.values())
        print([np.log(v) for v in I_tgt.values()])
        return ws, delta_ws
    check_inf = [np.isinf(v) for v in I_tgt.values()]
    if np.sum(check_inf):
        # early stopping if we get inf
        print('Early stop due to inf in I_tgt')
        return ws, delta_ws
    # checking for nan and inf 
    check_nan = [np.isnan(v) for v in I_ref.values()]
    if np.sum(check_nan):
        # early stopping if we get nan
        print('Early stop due to nan in I_ref')
        return ws, delta_ws
    check_inf = [np.isinf(v) for v in I_ref.values()]
    if np.sum(check_inf):
        # early stopping if we get inf
        print('Early stop due to inf in I_ref')
        return ws, delta_ws