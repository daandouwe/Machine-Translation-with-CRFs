import numpy as np
from sgd import sgd_minibatches
from collections import defaultdict
from processing import *
from features import featurize_edges, get_full_fset
from util import save_weights, load_weights, partition
from predict import predict


savepath = '../parses/eps-40k-ml10-3trans/'
predictpath =  'prediction/eps-40k-ml10-3trans/second/minibatch=10/'

parses = [load_parses_separate(savepath, k) for k in range(28000)]

# Optional: training on parses with non-empty ref-forests.
cleaned_parses = [(target_forest, ref_forest, src_fsa, tgt_sent) for (target_forest, ref_forest, src_fsa, tgt_sent) in parses if ref_forest]
print(len(cleaned_parses))

lexicon = load_lexicon(savepath)
fset = load_featureset(savepath)

# initialize weights uniformly
w_init = defaultdict(float)
for feature in fset:
    w_init[feature] = 1e-2

k = 10
minibatches = partition(cleaned_parses, k)
w_trained, delta_ws, likelihoods = sgd_minibatches(iters=5, delta_0=10, w=w_init, minibatches=minibatches, batch_size=k, parses=cleaned_parses, 
									  			   shuffle=True, sparse=True, scale_weight=2, regularizer=1, lmbda=1,
									  			   bar=True, log=False, log_last=False, check_convergence=False, 
									  			   savepath=False, prediction=predictpath, prediction_length=20)

print(likelihoods)

# printing for verification
w = w_trained[-1]
for k, v in sorted(w.items(), key=lambda x: x[1], reverse=True):
	print('{}'.format(k).ljust(25) + '{}'.format(v))