import pickle
from util import load_weights

savepath = '../parses/eps-200/trained-'

w = load_weights(savepath)[-1]

# for k in sorted(w.keys()):
# 	print('{}'.format(k).ljust(25) + '{}'.format(w[k]))

for k, v in sorted(w.items(), key=lambda x: x[1], reverse=True):
	print('{}'.format(k).ljust(25) + '{}'.format(v))
