from util import save_weights, load_weights
from processing import *
from graph import *
from features import weight_function


loadpath = 'data/dev1.zh-en'
lengthpath = 'dev123_lengths/dev1.zh-en.en.lengths'
savepath = 'prediction/'
weightpath = '../parses/eps-100/trained-'
parsepath = '../parses/eps-100/'

parses = [load_parses_separate(parsepath, k) for k in range(45)]

def predict(parses):
	
	w = load_weights(weightpath)[-1]
	f = open(savepath + 'viterbi-predictions.txt', 'w')
	g = open(savepath + 'sampled-predictions.txt', 'w')

	for parse in parses:
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
		
		n = 100
		d, count = sample(n, target_forest, tgt_tsort, tgt_edge_weights, I_tgt, root_tgt) # use exp!
		candidates = write_derrivation(d)
		sampled_translation = candidates.pop()

		f.write(viterbi_translation + '\n')
		g.write(sampled_translation + '\n')

	f.close()
	g.close()


predict(parses)

		
