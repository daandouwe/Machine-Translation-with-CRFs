from util import save_weights, load_weights, joint_prob
from processing import *
from graph import *
from features import weight_function
import progressbar
import libitg

def write_derrivation(d):
	derivation_as_fsa = libitg.forest_to_fsa(CFG(d), d[0].lhs)
	candidates = libitg.enumerate_paths_in_fsa(derivation_as_fsa)
	return candidates

parsepath = '../parses/eps-40k-ml10-3trans/'
k = 3


tgt_forest, ref_forest, src_fsa, tgt_sent = load_parses_separate(parsepath, k)

candidates = list(write_derrivation(list(tgt_forest)))
print(candidates[0])
print(candidates[0][-1])
correct_candidates = [sent for sent in candidates if sent[-1]=='?']

f = open('prediction/all-derivations-{}.txt'.format(k), 'w')
f.write(str(len(candidates)) + '\n\n' + '\n'.join(sorted(list(candidates))) + '\n\n' + str(len(correct_candidates)) + '\n\n'  + '\n'.join(sorted(list(correct_candidates))))
f.close()
