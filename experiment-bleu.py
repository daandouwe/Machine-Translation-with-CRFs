import numpy as np
from sgd import sgd_minibatches_bleu
from collections import defaultdict
from processing import *
from features import featurize_edges, get_full_fset
from util import save_weights, load_weights, partition, save_likelihoods
from predict import predict
import matplotlib.pyplot as plt


savepath = '../parses/eps-40k-ml10-3trans/'
predictpath =  'prediction/experiments/'

parses = [load_parses_separate(savepath, k) for k in range(100)]

# Optional: training on parses with non-empty ref-forests.
cleaned_parses = [(target_forest, ref_forest, src_fsa, tgt_sent) for (target_forest, ref_forest, src_fsa, tgt_sent) in parses if ref_forest][0:200]
lexicon = load_lexicon(savepath)
fset = load_featureset(savepath)

# initialize weights uniformly
w_init = defaultdict(float)
for feature in fset:
    w_init[feature] = 1e-2

k = 1
minibatches = partition(cleaned_parses, k)
bleu_step = 10

w_trained, delta_ws, likelihoods1, bleu1 = sgd_minibatches_bleu(iters=1, delta_0=10, w=w_init, minibatches=minibatches, batch_size=k, parses=cleaned_parses, 
									  			   			    shuffle=False, sparse=True, scale_weight=2, regularizer=False, lmbda=1,
									  			   			    bar=True, log=False, log_last=False, check_convergence=False, 
									  			   			    savepath=False, prediction=predictpath, prediction_length=20, bleu_step=bleu_step)
print(bleu1)

w_trained, delta_ws, likelihoods2, bleu2 = sgd_minibatches_bleu(iters=1, delta_0=10, w=w_init, minibatches=minibatches, batch_size=k, parses=cleaned_parses, 
									  			   			    shuffle=False, sparse=True, scale_weight=2, regularizer=False, lmbda=1,
									  			   			    bar=True, log=False, log_last=False, check_convergence=False, 
									  			   			    savepath=False, prediction=predictpath, prediction_length=20, bleu_step=bleu_step)
print(bleu2)

w_trained, delta_ws, likelihoods3, bleu3 = sgd_minibatches_bleu(iters=1, delta_0=10, w=w_init, minibatches=minibatches, batch_size=k, parses=cleaned_parses, 
									  			   				shuffle=False, sparse=True, scale_weight=2, regularizer=False, lmbda=1,
									  			   				bar=True, log=False, log_last=False, check_convergence=False, 
									  			   				savepath=False, prediction=predictpath, prediction_length=20, bleu_step=bleu_step)
print(bleu3)

# plotting bleu
handles = []
ax1 = plt.plot(range(1, bleu_step*len(bleu1)+1, bleu_step), list(zip(*bleu1))[0], '-', color='k', linewidth=0.8, label='100')
handles.extend(ax1)
ax2 = plt.plot(range(1, bleu_step*len(bleu2)+1, bleu_step), list(zip(*bleu2))[0], '--', color='k', linewidth=0.8, label='10')
handles.extend(ax2)
ax3 = plt.plot(range(1, bleu_step*len(bleu3)+1, bleu_step), list(zip(*bleu3))[0], '-.' ,color='k', linewidth=0.8, label='1')
handles.extend(ax3)
plt.legend(handles=handles)
plt.savefig(predictpath + 'bleu.pdf')
plt.clf()

# plotting bleu 1
handles = []
ax1 = plt.plot(range(1, bleu_step*len(bleu1)+1, bleu_step), list(zip(*bleu1))[1], '-', color='k', linewidth=0.8, label='100')
handles.extend(ax1)
ax2 = plt.plot(range(1, bleu_step*len(bleu2)+1, bleu_step), list(zip(*bleu2))[1], '--', color='k', linewidth=0.8, label='10')
handles.extend(ax2)
ax3 = plt.plot(range(1, bleu_step*len(bleu3)+1, bleu_step), list(zip(*bleu3))[1], '-.' ,color='k', linewidth=0.8, label='0.1')
handles.extend(ax3)
plt.legend(handles=handles)
plt.savefig(predictpath + 'bleu1.pdf')
plt.clf()


# plotting bleu 1
handles = []
ax1 = plt.plot(range(1, bleu_step*len(bleu1)+1, bleu_step), list(zip(*bleu1))[2], '-', color='k', linewidth=0.8, label='10')
handles.extend(ax1)
ax2 = plt.plot(range(1, bleu_step*len(bleu2)+1, bleu_step), list(zip(*bleu2))[2], '--', color='k', linewidth=0.8, label='1')
handles.extend(ax2)
ax3 = plt.plot(range(1, bleu_step*len(bleu3)+1, bleu_step), list(zip(*bleu3))[2], '-.' ,color='k', linewidth=0.8, label='0.1')
handles.extend(ax3)
plt.legend(handles=handles)
plt.savefig(predictpath + 'bleu2.pdf')
plt.clf()

# names = ['likelihoods1', 'likelihoods2', 'likelihoods2']
# for k, l in enumerate([likelihoods1, likelihoods2, likelihoods2]):	
# 	f = open('prediction/experiment/{}.txt'.format(names[k]), 'w')
# 	f.write('\n'.join(map(str, l)))
# 	f.close()

# printing for verification
w = w_trained[-1]
for k, v in sorted(w.items(), key=lambda x: x[1], reverse=True):
	print('{}'.format(k).ljust(25) + '{}'.format(v))
