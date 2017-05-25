from util import save_weights, load_weights, joint_prob
from processing import *
from graph import *
from features import weight_function
import progressbar

def predict(parses, w, k, savepath, sample=False, scale_weights=False):
	
	if scale_weights:
		for l, v in w.items():
			w[l] = scale_weights * v
		f = open(savepath + 'viterbi-predictions-{0}-{1}x.txt'.format(k, scale_weights), 'w')
		if sample: g = open(savepath + 'sampled-predictions-{0}-{1}x.txt'.format(k, scale_weights), 'w')

	else:
		f = open(savepath + 'viterbi-predictions-{0}.txt'.format(k), 'w')
		p = open(savepath + 'viterbi-predictions-{0}-probs.txt'.format(k), 'w')
		if sample: 
			g = open(savepath + 'sampled-predictions-{0}.txt'.format(k), 'w')
			h = open(savepath + 'sampled-predictions-{0}-counts.txt'.format(k), 'w')

	print('predicting...')
	bar = progressbar.ProgressBar(max_value=len(parses))

	for l, parse in enumerate(parses):
		bar.update(l)
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
		prob = joint_prob(d, tgt_edge_weights, I_tgt, root_tgt)

		
		if sample: 
			d, count = ancestral_sample(sample, target_forest, tgt_tsort, tgt_edge_weights, I_tgt, root_tgt) # use exp!
			candidates = write_derrivation(d)
			sampled_translation = candidates.pop()

		if l==len(parses)-1: # not enter on last line, otherwise perl script crashes
			f.write(viterbi_translation)
			p.write(str(prob))
			if sample: 
				g.write(sampled_translation)
				h.write('{0}/{1}'.format(count, sample))
		else:
			f.write(viterbi_translation + '\n')
			p.write(str(prob) + '\n')
			if sample: 
				g.write(sampled_translation + '\n')
				h.write('{0}/{1}\n'.format(count, sample))

		bar.update(l+1)

	f.close()
	if sample: 
		g.close()
		h.close()

	bar.finish()

if __name__ == "__main__":
	
	weightpath = 'trained-weights/eps-40k-ml10-3trans/trained-1-'
	parsepath = '../parses/dev/ml10-5trans/'
	# parsepath = '../parses/eps-40k-ml10-5trans/'
	# savepath = 'prediction/eps-40k-ml10-3trans/'
	savepath =  'prediction/nonempty/'
	# savepath = 'prediction/dev/ml10-3trans/'

	w = load_weights(weightpath)
	parses = [load_parses_separate(parsepath, k) for k in range(200)]
	predict(parses, w, k=1, savepath=savepath, scale_weights=False, sample=False)


		
