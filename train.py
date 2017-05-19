import numpy as np
from sgd import sgd_minibatch, sgd_minibatches
from collections import defaultdict
from processing import *
from features import featurize_edges, get_full_fset
from util import save_weights, load_weights, partition

# savepath = '../parses/29-sents-2-translations-sparse/'
savepath = '../parses/eps/'

parses = [load_parses_separate(savepath, k) for k in range(969)]

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
    w_init[feature] = 1e-5

# # pre-train the bad w_init with very low learing rate
# w_first, delta_ws = sgd_minibatch(1, 1e-6, w_init, minibatch=parses, 
#                                   sparse=True, bar=True, log=False, log_last=False,
#                                   check_convergence=True, scale_weight=2)

# save_weights(w_first[-1], savepath)

# then continue training with w_first[-1] (w_first is a list) with a bigger learning rate
# w_first = load_weights(savepath)

# # printing for verification
# for k in sorted(w_first.keys()):
# 	print(('\t'.join(map(str,[k, w_first[k]]))).expandtabs(25))
# print('\n')

# partition the parses in minibatches each of size 15 (e.g.)
minibatches = partition(parses, 15)
w_trained, delta_ws = sgd_minibatches(10, 1e-8, w_init, minibatches=minibatches, 
                                      sparse=True, bar=True, log=False, log_last=True,
                                      check_convergence=True, scale_weight=1, regularizer=False)

# w_test, delta_ws = sgd_minibatch(3, 1e-4, w_first, minibatch=parses, 
#                                  sparse=True, bar=True, log=False, log_last=True,
#                                  check_convergence=True, scale_weight=1,
#                                  regularizer=1e4)

save_weights(w_trained, savepath + 'trained-')

# printing for verification
w = w_trained[-1]
for k in sorted(w.keys()):
	print('{}'.format(k).ljust(25) + '{}'.format(w[k]))


