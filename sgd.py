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
    Uses a regularizer.
    NOTE: not sure if correct. Perhaps + regularizer * wmap_l1norm instead of -?
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


def sgd_minibatch(iters, delta, w, minibatch=[], 
                  sparse=False, log=False, bar=True, 
                  prob_log=False, log_last=False,
                  check_convergence=False,
                  scale_weight=4,
                  regularizer=False,
                  savepath=False):
    """
    Performs stochastic gradient descent on the weights vector w
    on a minibatch = [parses_1,parses_2,...,parses_N]
    """  
    ws = []
    delta_ws = []
    for i in range(iters):
        
        print('Iteration {}'.format(i+1))

        delta_w = 0.0
        w_new = defaultdict(float)
        if bar and not (i==iters-1 and log_last): bar = progressbar.ProgressBar(max_value=len(minibatch))
        
        for k, parse in enumerate(minibatch):
            
            if bar and not (i==iters-1 and log_last): bar.update(k)
            
            target_forest, ref_forest, src_fsa, tgt_sent = parse

            
            ### D_n(x) ###

            tgt_edge2fmap, _ = featurize_edges(target_forest, src_fsa,
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

            ref_edge2fmap, _ = featurize_edges(ref_forest, src_fsa,
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
            w_step, d_w = update_w(w, expected_features_D_xy, expected_features_Dn_x, delta=delta, regularizer=regularizer)
            
            # print('\n')
            # for k in sorted(w_step.keys()):
            #     print('{}'.format(k).ljust(25) + '{}'.format(w_step[k]))
            # print('\n')

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
                d, count = sample(n, target_forest, tgt_tsort, tgt_edge_weights, I_tgt, root_tgt) # use exp!
                candidates = write_derrivation(d)
                print('Most sampled: {0}/{1}'.format(count, n))
                print("Best y = '{}'".format(candidates.pop()))
                print('P(y,d|x) = {}\n'.format(joint_prob(d, tgt_edge_weights, I_tgt, root_tgt, log=prob_log)))

            if bar and not (i==iters-1 and log_last): bar.update(k+1)

        if bar and not (i==iters-1 and log_last): bar.finish()

        # print('\n')
        # for k in sorted(w.keys()):
        #     print('{}'.format(k).ljust(25) + '{}'.format(w[k]))
        # print('\n')

        # hack: scale weights so that they are at most of the scale 10**scale_weight
        abs_max = max(map(abs, w_new.values()))
        for k, v in w_new.items():
            w_new[k] = v / 10**(int(np.log10(abs_max))+1 - scale_weight)

        w = w_new        
        ws.append(w)
        delta_ws.append(delta_w)
        if check_convergence:
            print('delta w = {}\n'.format(delta_w))

        if savepath:
            save_weights(w, savepath + 'trained-{}-'.format(i+1))

    return ws, delta_ws


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
                    prediction_lentgh=10):
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
            print('OK')
            delta_w = 0.0
            w_new = defaultdict(float)
            
            delta_k = delta_0 * (1 + delta_0*(lmbda*(i*len(minibatches)+k)))**(-1) # this is delta_k = delta_0 when k=0 and i=0
            
            learning_rates.append(delta_k)

            if bar and not (i==iters-1 and log_last): bar.update(k)

            for l, parse in enumerate(minibatch):
                print('okido')
                # unpack parse

                target_forest, ref_forest, src_fsa, tgt_sent = parse
                print(target_forest)
                
                ### D_n(x) ###

                tgt_edge2fmap, _ = featurize_edges(target_forest, src_fsa,
                                                   sparse_del=sparse, sparse_ins=sparse, sparse_trans=sparse)

                # recompute edge weights
                tgt_edge_weights = {edge: np.exp(weight_function(edge, tgt_edge2fmap[edge], w)) for edge in target_forest}
                print('computing inside and outside')
                # compute inside and outside
                tgt_tsort = top_sort(target_forest)
                print('target sorted')
                root_tgt = Nonterminal("D_n(x)")
                I_tgt = inside_algorithm(target_forest, tgt_tsort, tgt_edge_weights)
                print('target inside done')
                O_tgt = outside_algorithm(target_forest, tgt_tsort, tgt_edge_weights, I_tgt, root_tgt)
                print('target outside done')
                # compute expected features
                expected_features_Dn_x = expected_feature_vector(target_forest, I_tgt, O_tgt, tgt_edge2fmap)

                ### D(x,y) ###

                ref_edge2fmap, _ = featurize_edges(ref_forest, src_fsa,
                                                   sparse_del=sparse, sparse_ins=sparse, sparse_trans=sparse)
                print('recomputing edge weights')
                # recompute edge weights
                ref_edge_weights = {edge: np.exp(weight_function(edge, ref_edge2fmap[edge], w)) for edge in ref_forest}

                # compute inside and outside
                tsort = top_sort(ref_forest)
                root_ref = Nonterminal("D(x,y)")
                I_ref = inside_algorithm(ref_forest, tsort, ref_edge_weights)
                O_ref = outside_algorithm(ref_forest, tsort, ref_edge_weights, I_ref, root_ref)
                print('calculating expected features')
                # compute expected features
                expected_features_D_xy = expected_feature_vector(ref_forest, I_ref, O_ref, ref_edge2fmap)
                print('expected_features calculated: ', expected_features_D_xy)
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
                    d, count = sample(n, target_forest, tgt_tsort, tgt_edge_weights, I_tgt, root_tgt) # use exp!
                    candidates = write_derrivation(d)
                    print('Most sampled: {0}/{1}'.format(count, n))
                    print("Best y = '{}'".format(candidates.pop()))
                    print('P(y,d|x) = {}\n'.format(joint_prob(d, tgt_edge_weights, I_tgt, root_tgt, log=prob_log)))

            if bar and not (i==iters-1 and log_last): bar.update(k+1)
            
            # print('\n')
            # for k in sorted(w.keys()):
            #     print('{}'.format(k).ljust(25) + '{}'.format(w[k]))
            # print('\n')

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
            predict(parses[0:prediction_lentgh], w, i, prediction)

    return ws, delta_ws


##########################
# Not needed any longer? #
##########################


def update_w_log(wmap, expected_features_D_xy, expected_features_Dn_x, delta=0.1):
    w_new = defaultdict(float)
    delta_w = 0.0 # holds the sum of deltas
    for rule in chain(expected_features_D_xy, expected_features_Dn_x):
        for feature in chain(expected_features_D_xy[rule], expected_features_Dn_x[rule]):
            d_w = delta * (expected_features_D_xy[rule][feature] - 
                           expected_features_Dn_x[rule][feature])
            w_new[feature] = wmap[feature] + d_w
            delta_w += abs(d_w)
    return w_new, delta_w



def sgd_func_minibatch_log(iters, delta, w, minibatch=[], 
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
            tgt_edge_weights = {edge: weight_function(edge, tgt_edge2fmap[edge], w) for edge in target_forest}

            if bar: bar.update(2)

            # compute inside and outside
            tgt_tsort = top_sort(target_forest)
            root_tgt = Nonterminal("D_n(x)")
            if bar: bar.update(3)
            I_tgt = inside_algorithm_log(target_forest, tgt_tsort, tgt_edge_weights) # log version
            if bar: bar.update(4)
            O_tgt = outside_algorithm_log(target_forest, tgt_tsort, tgt_edge_weights, I_tgt, root_tgt) # log version
            if bar: bar.update(5)

            # compute expected features
            expected_features_Dn_x = expected_feature_vector_log(target_forest, I_tgt, O_tgt, tgt_edge2fmap)
            if bar: bar.update(6)


            ### D(x,y) ###

            ref_edge2fmap, _ = featurize_edges(ref_forest, src_fsa,
                                               sparse_del=sparse, sparse_ins=sparse, sparse_trans=sparse)
            if bar: bar.update(7)

            # recompute edge weights
            ref_edge_weights = {edge: weight_function(edge, ref_edge2fmap[edge], w) for edge in ref_forest} # log version

            if bar: bar.update(8)

            # compute inside and outside
            tsort = top_sort(ref_forest)
            root_ref = Nonterminal("D(x,y)")
            if bar: bar.update(9)
            I_ref = inside_algorithm_log(ref_forest, tsort, ref_edge_weights)
            if bar: bar.update(10)
            O_ref = outside_algorithm_log(ref_forest, tsort, ref_edge_weights, I_ref, root_ref)
            if bar: bar.update(11)

            # compute expected features
            expected_features_D_xy = expected_feature_vector_log(ref_forest, I_ref, O_ref, ref_edge2fmap)
            if bar: bar.update(12)

            # update w
            w_step, d_w = update_w_log(w, expected_features_D_xy, expected_features_Dn_x, delta=delta)
            if bar: bar.update(13)
            
            print('\n')
            for k in sorted(w_step.keys()):
                print('{}'.format(k).ljust(25) + '{}'.format(w_step[k]))
            print('\n')

            delta_w += d_w / len(minibatch)
            for feature, value in w_step.items():
                w_new[feature] += value / len(minibatch)
            
            if bar: bar.finish()
        
            
            # testing for nan and inf


            if log or i==iters-1:
                d = viterbi_log(target_forest, tgt_tsort, tgt_edge_weights, I_tgt, root_tgt) # use exp!
                candidates = write_derrivation(d)
                print("x = '{}'".format(src_fsa.sent))
                print("Best y = '{}'".format(candidates.pop()))
                print('P(y,d|x) = {}\n'.format(joint_prob_log(d, tgt_edge_weights, I_tgt, root_tgt)))

        # print('\n')
        # for k in sorted(w.keys()):
        #     print('{}'.format(k).ljust(25) + '{}'.format(w[k]))
        # print('\n')

        w = w_new        
        ws.append(w)
        delta_ws.append(delta_w)
        if check_convergence:
            print('delta w = {}\n'.format(delta_w))
    
    return ws, delta_ws