import numpy as np
from sgd import sgd_func_minibatch
from collections import defaultdict
from processing import *
from features import featurize_edges, get_full_fset
from util import save_weights, load_weights

# savepath = '../parses/29-sents-2-translations-sparse/'
savepath = '../parses/eps/'

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

# initialize weights uniformly
w_init = defaultdict(float)
for feature in fset:
    w_init[feature] = 1e-5

# pre-train the bad w_init with very low learing rate
w_first, delta_ws = sgd_func_minibatch(1, 1e-6, w_init, minibatch=parses[0:25], 
                                      sparse=True, bar=False, log=True, check_convergence=True)

save_weights(w_first[-1], savepath)

# then continue training with w_first[-1] (w_first is a list) with a higher learning rate
w_first = load_weights(savepath)

# total = 0
# for k, v in w_first.items():
# 	total += abs(w_first[k])
# 	# total += w_first[k]**2

maxi = min(w_first.values())
for k, v in w_first.items():
	# w_first[k] = v / 10**(len(str(int(maxi))) - 5)
	w_first[k] = v / 10**(int(log10(x))+1 - 5)

for k in sorted(w_first.keys()):
	print(('\t'.join(map(str,[k, w_first[k]]))).expandtabs(25))
print('\n')

w_test, delta_ws = sgd_func_minibatch(10, 1e-5, w_first, minibatch=parses[0:25], 
                                      sparse=True, bar=False, log=True, check_convergence=True)

w = w_test[-1]
for k in sorted(w.keys()):
	print('{}'.format(k).ljust(25) + '{}'.format(w[k]))
	# print(('\t'.join(map(str,[k, w[k]]))).expandtabs(25))


