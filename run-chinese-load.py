import libitg as libitg
import numpy as np
from sgd import sgd_func_minibatch
from collections import defaultdict
from processing import *
from features import featurize_edges, get_full_fset
from util import save_weights, load_weights

savepath = '../parses/29-sents-2-translations-sparse/'

parses = [load_parses_separate(savepath, k) for k in range(29)]

lexicon = load_lexicon(savepath)

fset = load_featureset(savepath)


print('number of features: {}\n'.format(len(fset)))
print('\n'.join(fset))
print('\n')
for k, v in lexicon.items():
	print(k,v)
print('\n')
print('corpus size: {}\n'.format(len(parses)))

# w_init = defaultdict(float)
# for feature in fset:
#     w_init[feature] = 1e-5*np.random.uniform()

# # pre-train the bad w_init with very low learning rate
# w_first, delta_ws = sgd_func_minibatch(1, 1e-9, w_init, minibatch=parses[0:3], 
#                                       sparse=True, bar=False, log=True, check_convergence=True)

# save_weights(w_first[-1], savepath)

w_first = load_weights(savepath)

# then continue training with w_first[-1] (w_first is a list) with a higher learning rate
w_test, delta_ws = sgd_func_minibatch(5, 1, w_first, minibatch=parses, 
                                      sparse=True, bar=False, log=True, check_convergence=True)
