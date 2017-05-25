import numpy as np
from sgd import sgd_minibatches
from collections import defaultdict
from processing import *
from features import featurize_edges, get_full_fset
from util import save_weights, load_weights, partition, save_likelihoods
from predict import predict
import matplotlib.pyplot as plt


savepath = '../parses/eps-40k-ml10-3trans/'
predictpath =  'prediction/experiments/minibatch=1/lmbda=0.01/'

parses = [load_parses_separate(savepath, k) for k in range(10000)]

# Optional: training on parses with non-empty ref-forests.
cleaned_parses = [(target_forest, ref_forest, src_fsa, tgt_sent) for (target_forest, ref_forest, src_fsa, tgt_sent) in parses if ref_forest][0:1000]
print(len(cleaned_parses))
lexicon = load_lexicon(savepath)
fset = load_featureset(savepath)

# initialize weights uniformly
w_init = defaultdict(float)
for feature in fset:
    w_init[feature] = 1e-2

k = 1
minibatches = partition(cleaned_parses, k)
w_trained, delta_ws, likelihoods1 = sgd_minibatches(iters=1, delta_0=10, w=w_init, minibatches=minibatches, batch_size=k, parses=cleaned_parses, 
									  			   shuffle=False, sparse=True, scale_weight=2, regularizer=False, lmbda=0.01,
									  			   bar=True, log=False, log_last=False, check_convergence=False, 
									  			   savepath=False, prediction=False, prediction_length=False)

print(likelihoods1)

w_trained, delta_ws, likelihoods2 = sgd_minibatches(iters=1, delta_0=1, w=w_init, minibatches=minibatches, batch_size=k, parses=cleaned_parses, 
									  			   shuffle=False, sparse=True, scale_weight=2, regularizer=False, lmbda=0.01,
									  			   bar=True, log=False, log_last=False, check_convergence=False, 
									  			   savepath=False, prediction=False, prediction_length=False)
print(likelihoods2)

w_trained, delta_ws, likelihoods3 = sgd_minibatches(iters=1, delta_0=0.1, w=w_init, minibatches=minibatches, batch_size=k, parses=cleaned_parses, 
									  			   shuffle=False, sparse=True, scale_weight=2, regularizer=False, lmbda=0.01,
									  			   bar=True, log=False, log_last=False, check_convergence=False, 
									  			   savepath=False, prediction=False, prediction_length=False)
print(likelihoods3)

# plotting
handles = []
ax1 = plt.plot(range(1, len(likelihoods1)+1), likelihoods1, '-', color='k', linewidth=0.8, label='100')
handles.extend(ax1)
ax2 = plt.plot(range(1, len(likelihoods2)+1), likelihoods2, '--', color='k', linewidth=0.8, label='10')
handles.extend(ax2)
ax3 = plt.plot(range(1, len(likelihoods3)+1), likelihoods3, '-.' ,color='k', linewidth=0.8, label='1')
handles.extend(ax3)
plt.legend(handles=handles)
plt.savefig(predictpath + 'likelihoods.pdf')

# # printing for verification
# w = w_trained[-1]
# for k, v in sorted(w.items(), key=lambda x: x[1], reverse=True):
# 	print('{}'.format(k).ljust(25) + '{}'.format(v))