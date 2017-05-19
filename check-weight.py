import pickle
from util import load_weights

savepath = '../parses/eps/trained-2-'
savepath10 = '../parses/eps/trained-10-'

w = load_weights(savepath)
w10 = load_weights(savepath)

for k in sorted(w.keys()):
	print('{}'.format(k).ljust(25) + '{}'.format(w10[k]))