import numpy as np
import pickle
from libitg import Symbol, Terminal, Nonterminal, Span
from libitg import Rule, CFG
from libitg import FSA
from features import *
from processing import *
from graph import *
from collections import defaultdict, deque
from itertools import chain
from util import write_derrivation, joint_prob

def update_w(wmap, expected_features_D_xy, expected_features_Dn_x, delta=0.1):
    w_new = defaultdict(float)
    delta_w = 0.0 # holds the sum of deltas
    for rule in chain(expected_features_D_xy, expected_features_Dn_x):
        for feature in chain(expected_features_D_xy[rule], expected_features_Dn_x[rule]):
            d_w = delta * (expected_features_D_xy[rule][feature] - 
                           expected_features_Dn_x[rule][feature])
            w_new[feature] = wmap[feature] + d_w
            delta_w += d_w
    return w_new, delta_w

def sgd_func_minibatch(iters, delta, w, minibatch=[], 
                       sparse=False, log=False, bar=True, 
                       prob_log=False, check_convergence=False):
    """
    Performs stochastic gradient descent on the weights vector w.
    """  
    ws = []
    delta_ws = []
    for i in range(iters):
        
        print('Iteration {}'.format(i+1))

        delta_w = 0.0
        w_new = defaultdict(float)
        
        for k, parse in enumerate(minibatch):
            
            target_forest, ref_forest, src_fsa = parse

            if bar: bar = progressbar.ProgressBar(max_value=13)
            if bar: bar.update(0)
            
            ### D_n(x) ###

            tgt_edge2fmap, _ = featurize_edges(target_forest, src_fsa,
                                               sparse_del=sparse, sparse_ins=sparse, sparse_trans=sparse)

            if bar: bar.update(1)

            # recompute edge weights
            tgt_edge_weights = {edge: np.exp(weight_function(edge, tgt_edge2fmap[edge], w)) 
                                    for edge in target_forest}

            if bar: bar.update(2)

            # compute inside and outside
            tgt_tsort = top_sort(target_forest)
            root_tgt = Nonterminal("D_n(x)")
            if bar: bar.update(3)
            I_tgt = inside_algorithm(target_forest, tgt_tsort, tgt_edge_weights)
            if bar: bar.update(4)
            O_tgt = outside_algorithm(target_forest, tgt_tsort, tgt_edge_weights, I_tgt, root_tgt)
            if bar: bar.update(5)

            # compute expected features
            expected_features_Dn_x = expected_feature_vector(target_forest, I_tgt, O_tgt, tgt_edge2fmap)
            if bar: bar.update(6)


            ### D(x,y) ###

            ref_edge2fmap, _ = featurize_edges(ref_forest, src_fsa,
                                               sparse_del=sparse, sparse_ins=sparse, sparse_trans=sparse)
            if bar: bar.update(7)

            # recompute edge weights
            ref_edge_weights = {edge: np.exp(weight_function(edge, ref_edge2fmap[edge], w))
                                for edge in ref_forest}

            if bar: bar.update(8)

            # compute inside and outside
            tsort = top_sort(ref_forest)
            root_ref = Nonterminal("D(x,y)")
            if bar: bar.update(9)
            I_ref = inside_algorithm(ref_forest, tsort, ref_edge_weights)
            if bar: bar.update(10)
            O_ref = outside_algorithm(ref_forest, tsort, ref_edge_weights, I_ref, root_ref)
            if bar: bar.update(11)

            # compute expected features
            expected_features_D_xy = expected_feature_vector(ref_forest, I_ref, O_ref, ref_edge2fmap)
            if bar: bar.update(12)

            # update w
            w_step, d_w = update_w(w, expected_features_D_xy, expected_features_Dn_x, delta=delta)
            if bar: bar.update(13)
                
            delta_w += d_w / len(minibatch)
            for feature, value in w_step.items():
                w_new[feature] += value / len(minibatch)
            
            if bar: bar.finish()
        
            
            # testing for nan and inf


            if log or i==iters-1:
                d = viterbi(target_forest, tgt_tsort, tgt_edge_weights, I_tgt, root_tgt) # use exp!
                candidates = write_derrivation(d)
                print("x = '{}'".format(src_fsa.sent))
                print("Best y = '{}'".format(candidates.pop()))
                print('P(y,d|x) = {}\n'.format(joint_prob(d, tgt_edge_weights, I_tgt, root_tgt, log=prob_log))) # use

        w = w_new        
        ws.append(w)
        delta_ws.append(delta_w)
        if check_convergence:
            print('delta w = {}\n'.format(delta_w))
    
    return ws, delta_ws
