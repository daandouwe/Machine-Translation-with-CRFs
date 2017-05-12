import libitg
from libitg import Symbol, Terminal, Nonterminal, Span
from libitg import Rule, CFG
from libitg import FSA
from collections import defaultdict
import numpy as np
from features import *
from processing import *
from inside_outside import *


ch_en, en_ch, full_en_ch, full_ch_en = preprocess_lexicon()
corpus = read_data()

ch_sent, en_sent = corpus[1]
print(ch_sent)
print(en_sent)
lexicon = make_lexicon(ch_sent, ch_en, en_ch)
#     lexicon = make_lexicon_ALT(ch_sent, en_sent)

print(lexicon)
src_fsa = libitg.make_fsa(ch_sent)
print(src_fsa)
src_cfg = libitg.make_source_side_itg(lexicon)
print(src_cfg)
forest = libitg.earley(src_cfg, src_fsa, 
                       start_symbol=Nonterminal('S'), 
                       sprime_symbol=Nonterminal("D(x)"))
print('forest: {}'.format(len(forest)))

projected_forest = libitg.make_target_side_itg(forest, lexicon)
print('projected forest: {}'.format(len(projected_forest)))
tgt_fsa = libitg.make_fsa(en_sent)
print(tgt_fsa)
ref_forest = libitg.earley(projected_forest, tgt_fsa, 
                           start_symbol=Nonterminal("D(x)"), 
                           sprime_symbol=Nonterminal('D(x,y)'),
                           eps_symbol=Nonterminal('-EPS-'))

# NOTE: if lexicon does not contain the right translations with respect to the observed sentence y
# then ref_forest will be empty! See example above where we replaced `chien` with `chat`.

print('ref forest: {}'.format(len(ref_forest)))
print('pass1')
length_fsa = libitg.LengthConstraint(len(en_sent), strict=False)
print('pass2')

target_forest = libitg.earley(projected_forest, length_fsa, 
                              start_symbol=Nonterminal("D(x)"), 
                              sprime_symbol=Nonterminal("D_n(x)"))
print('target forest: {}'.format(len(target_forest)))
print('pass3')

# target_forest_as_fsa = libitg.forest_to_fsa(target_forest, Nonterminal('D_n(x)'))    
# candidates = libitg.enumerate_paths_in_fsa(target_forest_as_fsa)  
# print(len(candidates))
# for candidate in sorted(candidates)[0:10]:
#     print(candidate)    


# D_n(x)
tsort = top_sort(target_forest)
edge_weights = defaultdict(lambda:1)
root = Nonterminal("D_n(x)")

edge2fmap, fset = featurize_edges(target_forest, src_fsa,
                                    sparse_del=False, sparse_ins=False, sparse_trans=False)

I = inside_algorithm(target_forest, tsort, edge_weights)
O = outside_algorithm(target_forest, tsort, edge_weights, I, root)

expected_features_Dn_x = expected_feature_vector(target_forest, I, O, edge2fmap)

# D(x,y)
tsort = top_sort(ref_forest)
edge_weights = defaultdict(lambda:1)
root = Nonterminal("D(x,y)")

edge2fmap, fset = featurize_edges(ref_forest, src_fsa,
                                    sparse_del=False, sparse_ins=False, sparse_trans=False)

I = inside_algorithm(ref_forest, tsort, edge_weights)
O = outside_algorithm(ref_forest, tsort, edge_weights, I, root)

expected_features_D_xy = expected_feature_vector(ref_forest, I, O, edge2fmap) 