from util import save_weights, load_weights
from processing import *
from graph import *
from features import weight_function
import progressbar

loadpath = 'data/dev1.zh-en'
savepath = 'prediction/2k/'
weightpath = '../parses/eps-2k/weights/lr-20/trained-5-'

# Parsepath should be set to the path of the parses of the chinese development sentences in dev1.zh-en, generated
# in the same way as the training sentences. Note: we no longer need the development sentence lenghts
# from the folder dev123_lengths! We are using the epsilon constraint now.

# parsepath = '../parses/eps-100/'
parsepath = '../parses/eps-2k/' 

def predict(parses, w, k, savepath, sample=False, scale_weights=False):
	
	if scale_weights:
		for k, v in w.items():
			w[k] = scale_weights * v

	f = open(savepath + 'viterbi-predictions-{}.txt'.format(k), 'w')
	if sample: g = open(savepath + 'sampled-predictions.txt', 'w')

	print('predicting...')
	bar = progressbar.ProgressBar(max_value=len(parses))

	for k, parse in enumerate(parses):
		bar.update(k)
		# unpack parse                
		target_forest, ref_forest, src_fsa, tgt_sent = parse


		### D_n(x) ###
		tgt_edge2fmap, _ = featurize_edges(target_forest, src_fsa,
										   sparse_del=True, sparse_ins=True, sparse_trans=True)

		# recompute edge weights
		tgt_edge_weights = {edge: np.exp(weight_function(edge, tgt_edge2fmap[edge], w)) for edge in target_forest}

		# compute inside and outside
		tgt_tsort = top_sort(target_forest)
		root_tgt = Nonterminal("D_n(x)")
		I_tgt = inside_algorithm(target_forest, tgt_tsort, tgt_edge_weights)
		O_tgt = outside_algorithm(target_forest, tgt_tsort, tgt_edge_weights, I_tgt, root_tgt)

		### D(x,y) ###
		ref_edge2fmap, _ = featurize_edges(ref_forest, src_fsa,
										   sparse_del=True, sparse_ins=True, sparse_trans=True)

		# recompute edge weights
		ref_edge_weights = {edge: np.exp(weight_function(edge, ref_edge2fmap[edge], w)) for edge in ref_forest}

		# compute inside and outside
		tsort = top_sort(ref_forest)
		root_ref = Nonterminal("D(x,y)")
		I_ref = inside_algorithm(ref_forest, tsort, ref_edge_weights)
		O_ref = outside_algorithm(ref_forest, tsort, ref_edge_weights, I_ref, root_ref)

		#### PREDICT ####
		d = viterbi(target_forest, tgt_tsort, tgt_edge_weights, I_tgt, root_tgt) # use exp!
		candidates = write_derrivation(d)
		viterbi_translation = candidates.pop()
		
		if sample: 
			n = 100
			d, count = sample(n, target_forest, tgt_tsort, tgt_edge_weights, I_tgt, root_tgt) # use exp!
			candidates = write_derrivation(d)
			sampled_translation = candidates.pop()

		if k==len(parses)-1: # not enter on last line, otherwise perl script crashes
			f.write(viterbi_translation)
			if sample: g.write(sampled_translation)
		else:
			f.write(viterbi_translation + '\n')
			if sample: g.write(sampled_translation + '\n')

		bar.update(k+1)

	f.close()
	if sample: g.close()

	bar.finish()

if __name__ == "__main__":

	w = load_weights(weightpath)
	parses = [load_parses_separate(parsepath, k) for k in range(100)]
	predict(parses, w, k=5, savepath='prediction/2k/')

		
