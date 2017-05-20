import numpy as np
from sgd import sgd_minibatch, sgd_minibatches
from collections import defaultdict
from processing import *
from features import featurize_edges, get_full_fset
from util import save_weights, load_weights, partition
from predict import predict


# savepath = '../parses/29-sents-2-translations-sparse/'
# savepath = '../parses/eps-100/'
savepath = '../parses/eps-2k/'
predictpath =  'prediction/2k/'

parses = [load_parses_separate(savepath, k) for k in range(60)]

lexicon = load_lexicon(savepath)

fset = load_featureset(savepath)


# print('number of features: {}\n'.format(len(fset)))
# print('\n'.join(sorted(list(fset))))
# print('\n')
# for k, v in lexicon.items():
# 	print(k,v)
# print('\n')
# print('corpus size: {}\n'.format(len(parses)))

# initialize weights uniformly
w_init = defaultdict(float)
for feature in fset:
    w_init[feature] = 1e-2


# partition the parses in minibatches each of size 15 (e.g.)
minibatches = partition(parses, 20)
w_trained, delta_ws = sgd_minibatches(30, 5, w_init, minibatches=minibatches, 
                                      sparse=True, bar=True, log=False, log_last=True,
                                      check_convergence=True, scale_weight=1, regularizer=False,
                                      lmbda=0.2, savepath=savepath, prediction=predictpath)

# printing for verification
w = w_trained[-1]
for k, v in sorted(w.items(), key=lambda x: x[1], reverse=True):
	print('{}'.format(k).ljust(25) + '{}'.format(v))
