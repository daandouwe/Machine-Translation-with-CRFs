import libitg as libitg
import numpy as np
from sgd import sgd_func_minibatch, sgd_func_minibatch_log
from collections import defaultdict
from processing import *
from features import featurize_edges, get_full_fset
from util import save_weights, load_weights, save_weights_log, load_weights_log

savepath = '../parses/29-sents-2-translations-sparse/'

parses = [load_parses_separate(savepath, k) for k in range(29)]

lexicon = load_lexicon(savepath)

fset = load_featureset(savepath)


print('number of features: {}\n'.format(len(fset)))
print('\n'.join(sorted(list(fset))))
print('\n')
for k, v in lexicon.items():
	print(k,v)
print('\n')
print('corpus size: {}\n'.format(len(parses)))

# # initialize weights uniformly
# w_init = defaultdict(float)
# for feature in fset:
#     w_init[feature] = 1e-5

# # pre-train the bad w_init with very low learing rate
# w_first, delta_ws = sgd_func_minibatch_log(1, 1e-7, w_init, minibatch=parses[0:3], 
# 										   sparse=True, bar=False, log=True, check_convergence=True)

# save_weights_log(w_first[-1], savepath)

# then continue training with w_first[-1] (w_first is a list) with a higher learning rate
w_first = load_weights_log(savepath)

for k in sorted(w_first.keys()):
	print(('\t'.join(map(str,[k, w_first[k]]))).expandtabs(25))
print('\n')

w_test, delta_ws = sgd_func_minibatch_log(4, 1e-5, w_first, minibatch=parses[0:3], 
										  sparse=True, bar=False, log=True, check_convergence=True)

w = w_test[-1]
for k in sorted(w.keys()):
	print('{}'.format(k).ljust(25) + '{}'.format(w[k]))


