import pickle
from util import load_weights

savepath1 = '../parses/eps-2k/weights/lr-20/trained-1-'
savepath2 = '../parses/eps-2k/weights/lr-20/trained-2-'
savepath3 = '../parses/eps-2k/weights/lr-20/trained-3-'

w1 = load_weights(savepath1)
w2 = load_weights(savepath2)
w3 = load_weights(savepath3)

def check(w):
	for k, v in sorted(w.items(), key=lambda x: x[1], reverse=True):
		print('{}'.format(k).ljust(25) + '{}'.format(v))

def compare(w1, w2):
	for k, v in sorted(w1.items(), key=lambda x: x[1], reverse=True):
		print('{}'.format(k).ljust(25) + '{}'.format(v))
		print('\t{}'.format(k).ljust(25) + '{}'.format(w2[k]))

def check_difference(w1, w2):
	for k, v in sorted(w1.items(), key=lambda x: x[1], reverse=True):
		print('{}'.format(k).ljust(25) + '{}'.format(v - w2[k]))

# check(w1)
# compare(w1, w2)
check_difference(w1, w3)