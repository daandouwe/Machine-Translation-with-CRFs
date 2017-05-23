import numpy as np
import pickle
import random
from lib.formal import Symbol, Terminal, Nonterminal, Span, Rule, CFG, FSA
from features import *
from processing import *
from graph import *
from collections import defaultdict, deque
from itertools import chain
from util import write_derrivation, joint_prob, joint_prob_log, save_weights
import progressbar
from predict import predict
import itertools
from util import partition


def update_w(wmap, expected_features_D_xy, expected_features_Dn_x, delta=0.1, regularizer=False):
    """
    Uses an optional regularizer.
    """
    w_new = defaultdict(float)
    delta_w = 0.0 # holds the sum of deltas

    wmap_l2norm = np.sqrt(sum(np.square(list(wmap.values()))))

    for rule in chain(expected_features_D_xy, expected_features_Dn_x):
        for feature in chain(expected_features_D_xy[rule], expected_features_Dn_x[rule]):
            if regularizer:
                d_w = delta * (expected_features_D_xy[rule][feature] - 
                               expected_features_Dn_x[rule][feature] -
                               regularizer * wmap_l2norm)
            else:
                d_w = delta * (expected_features_D_xy[rule][feature] - 
                               expected_features_Dn_x[rule][feature])
            

            w_new[feature] = wmap[feature] + d_w
            delta_w += abs(d_w)
    return w_new, delta_w


def sgd_minibatches(iters, delta_0, w, minibatches=[], parses=[], batch_size=20,
                    sparse=False, log=False, bar=True, 
                    prob_log=False, log_last=False,
                    check_convergence=False,
                    scale_weight=False,
                    regularizer=False,
                    lmbda=2.0,
                    savepath=False,
                    prediction=False,
                    shuffle=False,
                    prediction_length=10):
    """
    Performs stochastic gradient descent on the weights vector w on
    minibatches = [minibatch_1, minibatch_2,....,minibatch_N].

    We are decaying the learning rate after each minibatch. We follow the following rule
    from http://cilvr.cs.nyu.edu/diglib/lsml/bottou-sgd-tricks-2012.pdf section 5.2:

    delta_k = delta_0 * (1 + delta_0*lmbda*k)**(−1)

    where k is the index of the minibatch and delta_0 is the initial learning rate,
    and lmbda is another hyperparameter that controls the rate of decay.
    """ 

    ws = []
    delta_ws = []
    for i in range(iters):
        
        print('Iteration {0}/{1}'.format(i+1, iters))

        learning_rates = list()
        if bar and not (i==iters-1 and log_last): bar = progressbar.ProgressBar(max_value=len(minibatches))
            
        if shuffle:
            minibatches = partition(random.sample(parses, len(parses)), batch_size)

        for k, minibatch in enumerate(minibatches):
            delta_w = 0.0
            w_new = defaultdict(float)
            
            delta_k = delta_0 * (1 + delta_0*(lmbda*(i*len(minibatches)+k)))**(-1) # this is delta_k = delta_0 when k=0 and i=0
            
            learning_rates.append(delta_k)

            if bar and not (i==iters-1 and log_last): bar.update(k)

            for l, parse in enumerate(minibatch):
                # unpack parse

                target_forest, ref_forest, src_fsa, tgt_sent = parse
                
                ### D_n(x) ###

                tgt_edge2fmap, _ = featurize_edges(target_forest, src_fsa, tgt_sent=tgt_sent,
                                                   sparse_del=sparse, sparse_ins=sparse, sparse_trans=sparse)

                # recompute edge weights
                tgt_edge_weights = {edge: np.exp(weight_function(edge, tgt_edge2fmap[edge], w)) for edge in target_forest}
                # compute inside and outside
                tgt_tsort = top_sort(target_forest)
                root_tgt = Nonterminal("D_n(x)")
                I_tgt = inside_algorithm(target_forest, tgt_tsort, tgt_edge_weights)
                O_tgt = outside_algorithm(target_forest, tgt_tsort, tgt_edge_weights, I_tgt, root_tgt)
                # compute expected features
                expected_features_Dn_x = expected_feature_vector(target_forest, I_tgt, O_tgt, tgt_edge2fmap)

                ### D(x,y) ###

                ref_edge2fmap, _ = featurize_edges(ref_forest, src_fsa, tgt_sent=tgt_sent,
                                                   sparse_del=sparse, sparse_ins=sparse, sparse_trans=sparse)
                # recompute edge weights
                ref_edge_weights = {edge: np.exp(weight_function(edge, ref_edge2fmap[edge], w)) for edge in ref_forest}

                # compute inside and outside
                tsort = top_sort(ref_forest)
                root_ref = Nonterminal("D(x,y)")
                I_ref = inside_algorithm(ref_forest, tsort, ref_edge_weights)
                O_ref = outside_algorithm(ref_forest, tsort, ref_edge_weights, I_ref, root_ref)
                # compute expected features
                expected_features_D_xy = expected_feature_vector(ref_forest, I_ref, O_ref, ref_edge2fmap)
                # update w
                w_step, d_w = update_w(w, expected_features_D_xy, expected_features_Dn_x, delta=delta_k, regularizer=regularizer)
                
                # print('\n')
                # for k in sorted(w_step.keys()):
                #     print('{}'.format(k).ljust(25) + '{}'.format(w_step[k]))
                # print('\n')

                # the update is averaged over the minibatch
                delta_w += d_w / len(minibatch)
                for feature, value in w_step.items():
                    w_new[feature] += value / len(minibatch)

                if log or (i==iters-1 and log_last):
                    print("x = '{}'".format(src_fsa.sent))
                    print("y = '{}'".format(tgt_sent))
                    
                    print('Viterbi')
                    d = viterbi(target_forest, tgt_tsort, tgt_edge_weights, I_tgt, root_tgt) # use exp!
                    candidates = write_derrivation(d)
                    print("Best y = '{}'".format(candidates.pop()))
                    print('P(y,d|x) = {}'.format(joint_prob(d, tgt_edge_weights, I_tgt, root_tgt, log=prob_log)))
                    
                    n = 100
                    d, count = ancestral_sample(n, target_forest, tgt_tsort, tgt_edge_weights, I_tgt, root_tgt) # use exp!
                    candidates = write_derrivation(d)
                    print('Most sampled: {0}/{1}'.format(count, n))
                    print("Best y = '{}'".format(candidates.pop()))
                    print('P(y,d|x) = {}\n'.format(joint_prob(d, tgt_edge_weights, I_tgt, root_tgt, log=prob_log)))

            if bar and not (i==iters-1 and log_last): bar.update(k+1)

            # hack: scale weights so that they are at most of the scale 10**scale_weight
            if scale_weight:
                abs_max = max(map(abs, w_new.values()))
                for k, v in w_new.items():
                    w_new[k] = v / 10**(int(np.log10(abs_max))+1 - scale_weight)

            # update after each minibatch
            w = w_new        
            ws.append(w)
            delta_ws.append(delta_w)

        if bar and not (i==iters-1 and log_last): bar.finish()

        if savepath:
            save_weights(w, savepath + 'trained-{}-'.format(i+1))

        if check_convergence:
            print('delta w: {}\n'.format([ds / len(w.keys()) for ds in delta_ws]))
            print('Learning rates: {}'.format(learning_rates))

        if prediction and i%5==0: # save every 5 iterations
            predict(parses[0:prediction_length], w, i+1, prediction)

    return ws, delta_ws