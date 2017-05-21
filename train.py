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
# predictpath =  'prediction/2k/full-3/'
predictpath =  'prediction/2k/'

parses = [load_parses_separate(savepath, k) for k in range(20)]

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


k = 1
minibatches = partition(parses, k)
w_trained, delta_ws = sgd_minibatches(iters=6, delta_0=10, w=w_init, minibatches=minibatches, batch_size=k, parses=parses, shuffle=True,
                                      sparse=True, bar=True, log=False, log_last=True,
                                      check_convergence=True, scale_weight=False, regularizer=10.0,
                                      lmbda=1, savepath=savepath+'weights/full-3', prediction=predictpath)

# printing for verification
w = w_trained[-1]
for k, v in sorted(w.items(), key=lambda x: x[1], reverse=True):
	print('{}'.format(k).ljust(25) + '{}'.format(v))
