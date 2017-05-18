import numpy as np
from sgd import sgd_func_minibatch
from collections import defaultdict
from processing import *
from features import featurize_edges, get_full_fset
from util import save_weights, load_weights

# savepath = '../parses/29-sents-2-translations-sparse/'
savepath = '../parses/eps/'

parses = [load_parses_separate(savepath, k) for k in range(40)]

lexicon = load_lexicon(savepath)

fset = load_featureset(savepath)


# print('number of features: {}\n'.format(len(fset)))
# print('\n'.join(sorted(list(fset))))
# print('\n')
# for k, v in lexicon.items():
# 	print(k,v)
# print('\n')
# print('corpus size: {}\n'.format(len(parses)))

# # initialize weights uniformly
# w_init = defaultdict(float)
# for feature in fset:
#     w_init[feature] = 1e-5

# # pre-train the bad w_init with very low learing rate
# w_first, delta_ws = sgd_func_minibatch(1, 1e-6, w_init, minibatch=parses, 
#                                       sparse=True, bar=True, log=False, log_last=False,
#                                       check_convergence=True, scale_weight=2)

# save_weights(w_first[-1], savepath)

# then continue training with w_first[-1] (w_first is a list) with a higher learning rate
w_first = load_weights(savepath)

# for k in sorted(w_first.keys()):
# 	print(('\t'.join(map(str,[k, w_first[k]]))).expandtabs(25))
# print('\n')

for minibatch in partition(parses, 10)
	w_test, delta_ws = sgd_func_minibatch(3, 1e-4, w_first, minibatch=parses, 
	                                      sparse=True, bar=True, log=False, log_last=True,
	                                      check_convergence=True, scale_weight=2)

w = w_test[-1]
for k in sorted(w.keys()):
	print('{}'.format(k).ljust(25) + '{}'.format(w[k]))
	# print(('\t'.join(map(str,[k, w[k]]))).expandtabs(25))


